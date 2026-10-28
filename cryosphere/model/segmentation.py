import torch
import numpy as np


def compute_segmentation_prior(N_residues, N_segments, start_residue, device):
    """
    Computes the segmentation prior if "uniform" is set in the yaml file
    :param N_residues: integer, number of residues
    :param N_segments: integer, number of domains
    :param start_residue: integer, starting residue of the segments.
    :param device: str, device to use
    :return: dict of means and std for each prior over the parameters of the GMM.
    """
    bound_0 = N_residues / N_segments
    segmentation_means_mean = torch.tensor(np.array([bound_0 / 2 + i * bound_0 for i in range(N_segments)]), dtype=torch.float32,
                          device=device)[None, :]

    segmentation_means_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]

    segmentation_stds_mean = torch.tensor(np.ones(N_segments) * bound_0, dtype=torch.float32, device=device)[None, :]

    segmentation_stds_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]

    segmentation_proportions_mean = torch.tensor(np.ones(N_segments) * 0, dtype=torch.float32, device=device)[None, :]

    segmentation_proportions_std = torch.tensor(np.ones(N_segments), dtype=torch.float32, device=device)[None, :]

    segmentation_prior = {}
    segmentation_prior["means"] = {"mean":segmentation_means_mean, "std":segmentation_means_std}
    segmentation_prior["stds"] = {"mean":segmentation_stds_mean, "std":segmentation_stds_std}
    segmentation_prior["proportions"] = {"mean":segmentation_proportions_mean, "std":segmentation_proportions_std}

    return segmentation_prior


class Segmentation(torch.nn.Module):
	def __init__(self, segmentation_config, residues_indexes, residues_chain, device="cpu", tau_segmentation=0.05):
		"""
		Creates a GMM used for segmentation purposes.
		:param segmentation_config: dictionnary, containing, for each part of the protein we want to segment, a dictionnary
									 {"part_i":{"N_segm":x, "start_res":res, "end_res":res, "chain":chain_id}}
		:param residues_indexes: np.array of integer, of indexes of the residues
		:param residues_chain: np.array of indexes of the chain each residue belongs too.
		:param N_residues: integer, total number of residues in the protein
		:param device: torch device to use.
		:param tau_segmentation: float, used to anneal the probabilities of the GMM
		"""
		super(Segmentation, self).__init__()
		self.segmentation_config = segmentation_config
		self.segments_means_means = torch.nn.ParameterDict({})
		self.segments_means_stds = torch.nn.ParameterDict({})
		self.segments_stds_means = torch.nn.ParameterDict({})
		self.segments_stds_stds = torch.nn.ParameterDict({})
		self.segments_proportions_means = torch.nn.ParameterDict({})
		self.segments_proportions_stds = torch.nn.ParameterDict({})
		self.tau_segmentation = tau_segmentation
		self.residues_indexes = torch.tensor(residues_indexes, dtype=torch.float32, device=device)[:, None]
		self.residues_chain = residues_chain
		self.device = device
		self.N_residues = len(self.residues_chain)

		for part, part_config in segmentation_config.items():
			N_segments = part_config["N_segm"]
			if part_config.get("all_protein", False):
				assert len(segmentation_config) == 1, "If the whole protein is segmented, only one segmentation can be defined."
				start_res = 0
				end_res = len(self.residues_indexes) - 1
				N_res = len(self.residues_indexes)
			else:
				start_res = part_config["start_res"]
				end_res = part_config["end_res"]
				N_res = end_res - start_res + 1

			if "segmentation_start_values" not in part_config:
				#Initialize the segmentation in a uniform way
				bound_0 = N_res/N_segments
				self.segments_means_means[part]= torch.nn.Parameter(data=torch.tensor(np.array([start_res + bound_0/2 + i*bound_0 for i in range(N_segments)]), dtype=torch.float32, device=device)[None, :],
				                                          requires_grad=True)
				self.segments_means_stds[part] = torch.nn.Parameter(data= torch.tensor(np.ones(N_segments)*10.0, dtype=torch.float32, device=device)[None,:],
				                                        requires_grad=True)

				self.segments_stds_means[part] = torch.nn.Parameter(data= torch.tensor(np.ones(N_segments)*bound_0, dtype=torch.float32, device=device)[None,:],
				                                        requires_grad=True)

				self.segments_stds_stds[part] = torch.nn.Parameter(
				    data=torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :],
				    requires_grad=True)

				self.segments_proportions_means[part] = torch.nn.Parameter(
				    data=torch.tensor(np.ones(N_segments) * 0, dtype=torch.float32, device=device)[None, :],
				    requires_grad=True)

				self.segments_proportions_stds[part] = torch.nn.Parameter(
				    data=torch.tensor(np.ones(N_segments), dtype=torch.float32, device=device)[None, :],
				    requires_grad=True)

			else:
				#Otherwise take the definitions of the segments
				self.segments_means_means[part] = torch.nn.Parameter(data = torch.tensor(np.array(part_config["segmentation_start_values"]["means_means"]), 
													dtype=torch.float32, device=device), requires_grad=True)

				self.segments_means_stds[part] = torch.nn.Parameter(data = torch.tensor(np.array(part_config["segmentation_start_values"]["means_stds"]), 
													dtype=torch.float32, device=device), requires_grad=True)

				self.segments_stds_means[part] = torch.nn.Parameter(data = torch.tensor(np.array(part_config["segmentation_start_values"]["stds_means"]), 
													dtype=torch.float32, device=device), requires_grad=True)

				self.segments_stds_stds[part] = torch.nn.Parameter(data = torch.tensor(np.array(part_config["segmentation_start_values"]["stds_stds"]), 
													dtype=torch.float32, device=device), requires_grad=True)

				self.segments_proportions_means[part] = torch.nn.Parameter(data = torch.tensor(np.array(part_config["segmentation_start_values"]["proportions_means"]), 
													dtype=torch.float32, device=device), requires_grad=True)

				self.segments_proportions_stds[part] = torch.nn.Parameter(data = torch.tensor(np.array(part_config["segmentation_start_values"]["proportions_stds"]), 
													dtype=torch.float32, device=device), requires_grad=True)

			self.segmentation_prior = {}
			for part, part_config in segmentation_config.items():
				prior_config = part_config.get("segmentation_prior") or {}
				prior_keys = [f"{v}_{s}" for v in ["means", "stds", "proportions"] for s in ["means", "stds"]]
				#A prior is custom when its type says so, or when it gives prior values without a type.
				prior_type = prior_config.get("type", "custom" if any(k in prior_config for k in prior_keys) else "uniform")
				if prior_type == "uniform":
				#Create a prior with values taken uniformly
					self.segmentation_prior[part] = {}
					N_segments = part_config["N_segm"]
					if part_config.get("all_protein", False):
						assert len(segmentation_config) == 1, "If the whole protein is segmented, only one segmentation can be defined."
						start_res = 0
						end_res = len(self.residues_indexes) - 1
						N_res = len(self.residues_indexes)
					else:
						start_res = part_config["start_res"]
						end_res = part_config["end_res"]
						N_res = end_res - start_res + 1

					bound_0 = N_res / N_segments
					segmentation_means_mean = torch.tensor(np.array([start_res + bound_0 / 2 + i * bound_0 for i in range(N_segments)]), dtype=torch.float32,
					          device=device)[None, :]
					segmentation_means_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]
					segmentation_stds_mean = torch.tensor(np.ones(N_segments) * bound_0, dtype=torch.float32, device=device)[None, :]
					segmentation_stds_std = torch.tensor(np.ones(N_segments) * 10.0, dtype=torch.float32, device=device)[None, :]
					segmentation_proportions_mean = torch.tensor(np.ones(N_segments) * 0, dtype=torch.float32, device=device)[None, :]
					segmentation_proportions_std = torch.tensor(np.ones(N_segments), dtype=torch.float32, device=device)[None, :]
					self.segmentation_prior[part]["means"] = {"mean":segmentation_means_mean, "std":segmentation_means_std}
					self.segmentation_prior[part]["stds"] = {"mean":segmentation_stds_mean, "std":segmentation_stds_std}
					self.segmentation_prior[part]["proportions"] = {"mean":segmentation_proportions_mean, "std":segmentation_proportions_std}

				else:
					# Otherwise take the prior values input by the user, as tensors shaped like the uniform prior.
					missing = [k for k in prior_keys if k not in prior_config]
					if missing:
						raise ValueError(f"The custom segmentation_prior of {part} is missing {missing}.")
					self.segmentation_prior[part] = {}
					for type_value in ["means", "stds", "proportions"]:
						prior_values = {}
						for stat in ["mean", "std"]:
							values = np.array(prior_config[f"{type_value}_{stat}s"], dtype=np.float32).reshape(-1)
							if values.size == 1:
								#One value for all the segments.
								values = np.full(part_config["N_segm"], values[0], dtype=np.float32)
							if values.size != part_config["N_segm"]:
								raise ValueError(f"segmentation_prior {type_value}_{stat}s of {part} must have one value or N_segm = {part_config['N_segm']} values.")
							prior_values[stat] = torch.tensor(values, device=device)[None, :]
						self.segmentation_prior[part][type_value] = prior_values


	def sample_segmentation(self, N_batch, part_config, part):
		"""
		Samples a segmantion
		:param N_batch: integer: size of the batch.
		:param N_segments: integer, number of segments
		:param part_config: dictionnary, containing the parameters of the GMM for segmenting
		:param part: part of the protein we want to sample a segmentation for.
		:return: dictionnary of torch.tensor(N_batch, N_residues, N_segments) values of the segmentation, np.array of 0 and 1, 
				mask to get the residues to which we apply the segmentation, in the frame of the total protein, not of the chain.
		"""
		N_segments = part_config["N_segm"]
		if part_config.get("all_protein", False):
			residues_chain = self.residues_indexes
			mask = np.ones(self.N_residues, dtype=np.float32)
			start_res = 0
			end_res = len(residues_chain) - 1
		else:
			chain_id = part_config["chain"]
			#Be careful: the start and end residues are included and the residue numbering starts at 0.
			residues_chain = self.residues_indexes[self.residues_chain == chain_id]
			mask = np.zeros(self.N_residues, dtype=np.float32)
			tmp_array = mask[self.residues_chain == chain_id]
			tmp_array[[i for i in range(part_config["start_res"], part_config["end_res"]+1)]] = 1
			mask[self.residues_chain == chain_id] = tmp_array
			start_res = part_config["start_res"]
			end_res = part_config["end_res"]

		#In residues_chain, we have the indexes of the relevant residues in the frame of the total protein. We want to find their indexes in the frame of the chain, so we 
		# minus the first indexes of that chain
		residues = residues_chain[[i for i in range(start_res, end_res+1)]] - torch.min(residues_chain)
		#We sample the proportions of the GMM
		cluster_proportions = torch.randn((N_batch, N_segments),
		                                  device=self.device) * self.segments_proportions_stds[part] + self.segments_proportions_means[part] 
		#We sample the cluster means
		cluster_means = torch.randn((N_batch, N_segments), device=self.device) * self.segments_means_stds[part] + self.segments_means_means[part] 
		#We sample the cluster stds: elu(x) + 1, computed as exp(x) for x <= 0 because in float32 elu(x) + 1 cancels to exactly 0
		#for x < -17.4, and a zero std makes the gradients of the segmentation NaN. The floor keeps the std strictly positive.
		std_logits = torch.randn((N_batch, N_segments), device=self.device)*self.segments_stds_stds[part]  + self.segments_stds_means[part]
		cluster_std = torch.clamp(torch.where(std_logits > 0, std_logits + 1, torch.exp(torch.clamp(std_logits, max=0))), min=1e-2)
		#We take the log of the softmax of the proportions, to have them positive and summing to one
		log_proportions = torch.log_softmax(cluster_proportions, dim=-1)
		log_num = -0.5*(residues[None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
		      log_proportions[:, None, :]

		return {"segmentation":torch.softmax(log_num / self.tau_segmentation, dim=-1), "mask":mask}

	def sample_segments(self, N_batch):
		"""
		Function sampling a segmentation based on the current parameters of the segmentation.
		:param N_batch: integer, batch_size
		:return: all_segmentations, dictionnary containing, for each part we want to segment, the values of the stochastic matrix and the residue indexes it is applied to.
		"""
		all_segmentations = {}
		for part, part_config in self.segmentation_config.items():
			segmentation = self.sample_segmentation(N_batch, part_config, part)
			all_segmentations[part] = segmentation

		return all_segmentations














