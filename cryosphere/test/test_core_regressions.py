"""
Regression tests for numerical and crash bugs: they run on CPU with small synthetic inputs (no dataset, no GPU).
Run from the repository root with: python -m pytest cryosphere/test/test_core_regressions.py
(running the file directly would import an installed cryosphere instead of this checkout).
"""
import os
import logging
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
import yaml

from cryosphere.model import utils
from cryosphere.model.ctf import CTF
from cryosphere.model.dataset import ImageDataSet
from cryosphere.model.gmm import Gaussian
from cryosphere.model.loss import compute_KL_prior_segments, compute_all_beta_schedule, compute_clashing_distances
from cryosphere.model.mlp import MLP
from cryosphere.model.polymer import Polymer
from cryosphere.model.segmentation import Segmentation
from cryosphere.model.vae import VAE

REPO_ROOT = Path(__file__).resolve().parents[2]
NPIX, APIX = 16, 2.0


def write_ca_pdb(path, chains):
    """Write a C-alpha only model: chains is a list of (chain_id, number_of_residues), residues 3.8 Å apart."""
    lines, k = [], 0
    for c, (chain, n) in enumerate(chains):
        for r in range(n):
            x, y, z = 3.8 * r - 1.9 * n, 6.0 * c, 0.0
            k += 1
            lines.append("ATOM  %5d  CA  ALA %s%4d    %8.3f%8.3f%8.3f  1.00  0.00           C" % (k, chain, r + 1, x, y, z))
    Path(path).write_text("\n".join(lines) + "\nEND\n")


@contextmanager
def own_root_logger():
    """parse_yaml configures the root logger with basicConfig: give it a fresh one and restore the caller's afterwards."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    try:
        yield
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def write_star(path, n_particles, stack_name="particles.mrcs", drop=()):
    optics = pd.DataFrame({"rlnOpticsGroup": [1], "rlnVoltage": [300.0], "rlnSphericalAberration": [2.7],
                           "rlnAmplitudeContrast": [0.1], "rlnImagePixelSize": [APIX], "rlnImageSize": [NPIX]})
    rng = np.random.default_rng(0)
    particles = pd.DataFrame({"rlnImageName": [f"{i + 1:06d}@{stack_name}" for i in range(n_particles)],
                              "rlnAngleRot": rng.uniform(-180, 180, n_particles),
                              "rlnAngleTilt": rng.uniform(0, 180, n_particles),
                              "rlnAnglePsi": rng.uniform(-180, 180, n_particles),
                              "rlnOriginXAngst": np.zeros(n_particles), "rlnOriginYAngst": np.zeros(n_particles),
                              "rlnDefocusU": rng.uniform(10000, 20000, n_particles),
                              "rlnDefocusV": rng.uniform(10000, 20000, n_particles),
                              "rlnDefocusAngle": rng.uniform(0, 180, n_particles),
                              "rlnOpticsGroup": np.ones(n_particles, dtype=int)})
    particles = particles.drop(columns=list(drop))
    starfile.write({"optics": optics, "particles": particles}, path, overwrite=True)


def write_stack(path, n_particles):
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.random.default_rng(1).normal(size=(n_particles, NPIX, NPIX)).astype(np.float32))
        m.voxel_size = APIX


class TestCTFFromStarfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.star = os.path.join(self.tmp.name, "particles.star")

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_phase_shift_defaults_to_zero(self):
        write_star(self.star, 10)
        ctf = CTF.create_ctf({"file": self.star}, apix=APIX, side_shape=NPIX)
        self.assertTrue(torch.all(ctf.phaseShift == 0))
        self.assertTrue(torch.isfinite(ctf.compute_ctf(torch.arange(10))).all())

    def test_phase_shift_in_optics_block_is_used(self):
        write_star(self.star, 10)
        star = starfile.read(self.star, always_dict=True)
        star["optics"]["rlnPhaseShift"] = 90.0
        starfile.write(star, self.star, overwrite=True)
        ctf = CTF.create_ctf({"file": self.star}, apix=APIX, side_shape=NPIX)
        self.assertTrue(torch.all(ctf.phaseShift == 90))

    def test_missing_required_column_raises(self):
        write_star(self.star, 10, drop=("rlnDefocusU",))
        with self.assertRaisesRegex(ValueError, "rlnDefocusU"):
            CTF.create_ctf({"file": self.star}, apix=APIX, side_shape=NPIX)


class TestSegmentation(unittest.TestCase):
    def make(self, config, chains):
        chain_ids = np.concatenate([[c] * n for c, n in chains])
        return Segmentation(config, np.arange(len(chain_ids)), chain_ids)

    def test_segment_width_does_not_underflow(self):
        """elu(x) + 1 is exactly 0 in float32 for x < -17.4; the resulting zero width made the gradients NaN."""
        seg = self.make({"part1": {"all_protein": True, "N_segm": 4}}, [("A", 200)])
        with torch.no_grad():
            seg.segments_stds_means["part1"][0, 0] = -40.0
        out = seg.sample_segments(8)["part1"]["segmentation"]
        (out * torch.randn_like(out)).sum().backward()
        self.assertTrue(torch.isfinite(out).all())
        for name, p in seg.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_extreme_proportion_logits_keep_gradients_finite(self):
        """softmax underflows to 0 for logit gaps above ~104 in float32, and log(0) = -inf made the gradients NaN."""
        seg = self.make({"part1": {"all_protein": True, "N_segm": 4}}, [("A", 200)])
        with torch.no_grad():
            seg.segments_proportions_means["part1"].copy_(torch.tensor([[200.0, -200.0, 0.0, 0.0]]))
            seg.segments_proportions_stds["part1"].zero_()
        out = seg.sample_segments(8)["part1"]["segmentation"]
        (out * torch.randn_like(out)).sum().backward()
        for name, p in seg.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_segment_width_unchanged_in_normal_range(self):
        """Away from the underflow the sampled segmentation is still the one given by elu(x) + 1."""
        torch.manual_seed(0)
        seg = self.make({"part1": {"all_protein": True, "N_segm": 4}}, [("A", 100)])
        with torch.no_grad():
            for d in (seg.segments_means_stds, seg.segments_stds_stds, seg.segments_proportions_stds):
                d["part1"].zero_()
            seg.segments_stds_means["part1"].copy_(torch.tensor([[-3.0, 0.5, 5.0, 40.0]]))
            seg.segments_proportions_means["part1"].copy_(torch.tensor([[0.1, -0.2, 0.3, 0.0]]))
        out = seg.sample_segments(1)["part1"]["segmentation"][0].double()
        mu = seg.segments_means_means["part1"].detach().double()
        sd = torch.nn.functional.elu(seg.segments_stds_means["part1"].detach().double()) + 1
        log_p = torch.log_softmax(seg.segments_proportions_means["part1"].detach().double(), -1)
        r = torch.arange(100, dtype=torch.float64)[:, None]
        expected = torch.softmax((-0.5 * (r - mu) ** 2 / sd ** 2 + log_p) / seg.tau_segmentation, dim=-1)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_custom_prior_without_type(self):
        config = {"part1": {"start_res": 0, "end_res": 99, "chain": "A", "N_segm": 3,
                            "segmentation_prior": {"means_means": [10, 50, 90], "means_stds": [5, 5, 5],
                                                   "stds_means": [15, 15, 15], "stds_stds": [5, 5, 5],
                                                   "proportions_means": [0, 0, 0], "proportions_stds": [1, 1, 1]}}}
        seg = self.make(config, [("A", 120)])
        self.assertEqual(tuple(seg.segmentation_prior["part1"]["means"]["mean"].shape), (1, 3))
        for variable in ("means", "stds", "proportions"):
            kl = compute_KL_prior_segments(seg, seg.segmentation_prior, variable, "1e-10")
            self.assertTrue(torch.isfinite(kl))

    def test_custom_prior_wrong_length_raises(self):
        config = {"part1": {"start_res": 0, "end_res": 99, "chain": "A", "N_segm": 3,
                            "segmentation_prior": {"means_means": [10, 50], "means_stds": [5, 5],
                                                   "stds_means": [15, 15], "stds_stds": [5, 5],
                                                   "proportions_means": [0, 0], "proportions_stds": [1, 1]}}}
        with self.assertRaisesRegex(ValueError, "means_means"):
            self.make(config, [("A", 120)])

    def test_custom_prior_with_missing_key_raises(self):
        config = {"part1": {"start_res": 0, "end_res": 99, "chain": "A", "N_segm": 3,
                            "segmentation_prior": {"type": "custom", "mean_means": [10, 50, 90], "means_stds": [5, 5, 5],
                                                   "stds_means": [15, 15, 15], "stds_stds": [5, 5, 5],
                                                   "proportions_means": [0, 0, 0], "proportions_stds": [1, 1, 1]}}}
        with self.assertRaisesRegex(ValueError, "means_means"):
            self.make(config, [("A", 120)])

    def test_custom_prior_with_one_value_for_all_segments(self):
        config = {"part1": {"start_res": 0, "end_res": 99, "chain": "A", "N_segm": 3,
                            "segmentation_prior": {"type": "custom", "means_means": 50.0, "means_stds": 5.0, "stds_means": 15.0,
                                                   "stds_stds": 5.0, "proportions_means": 0.0, "proportions_stds": 1.0}}}
        seg = self.make(config, [("A", 120)])
        self.assertTrue(torch.equal(seg.segmentation_prior["part1"]["means"]["mean"], torch.full((1, 3), 50.0)))
        for variable in ("means", "stds", "proportions"):
            self.assertTrue(torch.isfinite(compute_KL_prior_segments(seg, seg.segmentation_prior, variable, "1e-10")))

    def test_example_yaml_files_load_and_build(self):
        examples = {"parameters.yaml": [("A", 400)], "parameters_two_segmentation.yaml": [("A", 600), ("C", 400)]}
        for name, chains in examples.items():
            path = REPO_ROOT / name
            if not path.exists():
                self.skipTest(f"{name} is only in a source checkout")
            settings = yaml.safe_load(path.read_text())
            self.assertIsInstance(settings["epsilon_kl"], str)
            seg = self.make(settings["segmentation_config"], chains)
            for variable in ("means", "stds", "proportions"):
                kl = compute_KL_prior_segments(seg, seg.segmentation_prior, variable, settings["epsilon_kl"])
                self.assertTrue(torch.isfinite(kl), name)


class TestLosses(unittest.TestCase):
    def test_full_clashing_loss_without_clashes_is_zero(self):
        line = torch.tensor([[5.0 * i, 0.0, 0.0] for i in range(20)])[None].repeat(3, 1, 1)
        self.assertEqual(compute_clashing_distances(line, "cpu", cutoff=4).item(), 0.0)
        clashing = line.clone()
        clashing[:, 10] = clashing[:, 0] + 1.0
        self.assertGreater(compute_clashing_distances(clashing, "cpu", cutoff=4).item(), 0.0)

    def test_default_schedules_with_few_epochs(self):
        losses = {"KL_prior_latent": {"schedule": "linear"}, "KL_prior_segmentation_proportions": {"schedule": "cyclical"}}
        for n_epochs in (1, 2, 5):
            for epoch in range(n_epochs):
                betas = compute_all_beta_schedule(epoch, n_epochs, losses)
                self.assertTrue(all(0 <= b <= 1 for b in betas.values()))


class TestDataAndLogs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset_with_fewer_than_100_particles(self):
        write_star(self.dir / "particles.star", 20)
        write_stack(self.dir / "particles.mrcs", 20)
        dataset = ImageDataSet(APIX, NPIX, {"file": str(self.dir / "particles.star")}, str(self.dir))
        self.assertEqual(len(dataset), 20)
        self.assertTrue(np.isfinite(dataset.f_std))

    def write_experiment(self, n):
        write_star(self.dir / "particles.star", n)
        write_stack(self.dir / "particles.mrcs", n)
        write_ca_pdb(self.dir / "base.pdb", [("A", 30)])
        (self.dir / "images.yaml").write_text(yaml.safe_dump({"apix": APIX, "Npix": NPIX, "Npix_downsize": NPIX, "sigma_gmm": 2}))
        settings = {"name": "test", "wandb": False, "folder_path": str(self.dir), "base_structure_path": "base.pdb",
                    "image_yaml": "images.yaml", "particles_path": str(self.dir),
                    "cs_star_file": {"file": str(self.dir / "particles.star")},
                    "segmentation_config": {"part1": {"all_protein": True, "N_segm": 2}}, "lp_bandwidth": None,
                    "input_mask_radius": 1, "latent_dimension": 2, "tau_segmentation": 0.05, "device": "cpu",
                    "num_workers": 0, "amortized": True, "N_images": n, "N_epochs": 2, "batch_size": 10,
                    "epsilon_kl": "1e-10", "seed": 1, "deterministic_cuda": False,
                    "encoder": {"hidden_dimensions": [8]}, "decoder": {"hidden_dimensions": [8]},
                    "optimizer": {"name": "adam", "learning_rate": 1e-4},
                    "loss": {"clashing_loss": {"full_clashing_loss": True, "clashing_cutoff": 4, "schedule": "constant", "beta": 0.1}},
                    "resume_training": {"model": None, "segmentation": None}}
        (self.dir / "parameters.yaml").write_text(yaml.safe_dump(settings))
        return self.dir / "parameters.yaml"

    def test_analysis_does_not_overwrite_training_log(self):
        parameters = self.write_experiment(100)     # enough particles to avoid the small-dataset path
        (self.dir / "cryoSPHERE").mkdir()
        (self.dir / "cryoSPHERE" / "run.log").write_text("training log\n")
        with own_root_logger():
            utils.parse_yaml(str(parameters), 0, analyze=True)
        self.assertEqual((self.dir / "cryoSPHERE" / "run.log").read_text(), "training log\n")
        self.assertTrue((self.dir / "cryoSPHERE" / "analysis.log").exists())

    def test_training_still_logs_to_run_log(self):
        parameters = self.write_experiment(100)
        with own_root_logger():
            utils.parse_yaml(str(parameters), 1, analyze=False)      # gpu_id 1 skips the W&B login
        self.assertIn("Running cryoSPHERE", (self.dir / "cryoSPHERE" / "run.log").read_text())
        self.assertFalse((self.dir / "cryoSPHERE" / "analysis.log").exists())


class TestPCAAnalysis(unittest.TestCase):
    def setUp(self):
        from cryosphere.data import analyze
        self.analyze = analyze
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_ca_pdb(self.dir / "base.pdb", [("A", 30)])
        self.base = Polymer.from_pdb(str(self.dir / "base.pdb"))
        coords = torch.tensor(self.base.coord, dtype=torch.float32)
        self.gmm = Gaussian(coords, torch.ones((len(coords), 1)) * 2, torch.ones((len(coords), 1)) * 6)
        self.config = {"part1": {"all_protein": True, "N_segm": 2}}
        self.segmenter = Segmentation(self.config, np.arange(len(coords)), self.base.chain_id)

    def tearDown(self):
        self.tmp.cleanup()

    def vae(self, latent_dim):
        encoder = MLP(NPIX ** 2, latent_dim * 2, [8], network_type="encoder", device="cpu")
        decoder = MLP(latent_dim, 2 * 6, [8], network_type="decoder", device="cpu")
        return VAE(encoder, decoder, "cpu", self.config, latent_dim=latent_dim, amortized=True)

    def test_traversals_for_dimensions_not_starting_at_zero(self):
        z = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
        with torch.no_grad():
            self.analyze.run_pca_analysis(self.vae(3), z, [1, 2], 3, str(self.dir), self.gmm, self.base, 1, self.segmenter, "cpu")
        for dim in (1, 2):
            self.assertTrue((self.dir / f"pc{dim}" / "pca.png").exists())
            self.assertEqual(len(list((self.dir / f"pc{dim}").glob("structure_z_*.pdb"))), 3)

    def test_traversal_for_one_dimensional_latent_space(self):
        z = np.random.default_rng(0).normal(size=(50, 1)).astype(np.float32)
        with torch.no_grad():
            self.analyze.run_pca_analysis(self.vae(1), z, [0], 3, str(self.dir), self.gmm, self.base, 1, self.segmenter, "cpu")
        self.assertEqual(len(list((self.dir / "pc0").glob("structure_z_*.pdb"))), 3)


class TestCenterOrigin(unittest.TestCase):
    def test_structure_file_formats(self):
        """mmCIF input used an undefined PDBxFile; extensions are matched case-insensitively."""
        from biotite.structure.io.pdb import PDBFile
        from biotite.structure.io.pdbx import CIFFile, set_structure
        from cryosphere.data.center_origin import center_origin
        with tempfile.TemporaryDirectory() as tmp:
            write_ca_pdb(os.path.join(tmp, "source.pdb"), [("A", 10)])
            atoms = PDBFile.read(os.path.join(tmp, "source.pdb")).get_structure(model=1)
            cif = CIFFile()
            set_structure(cif, atoms)
            cif.write(os.path.join(tmp, "source.cif"))
            with mrcfile.new(os.path.join(tmp, "map.mrc")) as m:
                m.set_data(np.zeros((NPIX,) * 3, dtype=np.float32))
                m.voxel_size = APIX
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                for name, source in [("model.cif", "source.cif"), ("model.CIF", "source.cif"), ("model.mmcif", "source.cif"),
                                     ("model.ent", "source.pdb"), ("model.PDB", "source.pdb")]:
                    shutil.copyfile(source, name)
                    center_origin(name, "map.mrc")
                    centered = PDBFile.read("model_centered.pdb").get_structure(model=1)
                    np.testing.assert_allclose(centered.coord, atoms.coord - NPIX // 2 * APIX, atol=1e-3, err_msg=name)
                    os.remove("model_centered.pdb")
            finally:
                os.chdir(cwd)


class TestDDPSetup(unittest.TestCase):
    def setup_env(self, env):
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(utils.torch.cuda, "set_device"), mock.patch.object(utils, "init_process_group"):
            for key in [k for k in ("MASTER_PORT",) if k not in env]:
                os.environ.pop(key, None)
            utils.ddp_setup(0, 1)
            return os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"]

    def test_default_port(self):
        self.assertEqual(self.setup_env({}), ("localhost", "12355"))

    def test_port_from_environment(self):
        self.assertEqual(self.setup_env({"MASTER_PORT": "23456", "MASTER_ADDR": "some.other.host"}), ("localhost", "23456"))
