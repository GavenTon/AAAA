from math import pi
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from torch.distributions import Normal


class cluster_GMM_Dist:
    def __init__(
        self, clustering_model, vars, method="kmeans", normalize_direction=False
    ) -> None:
        if method == "kmeans":
            self.clustering_model = clustering_model
            self.num_clusters = clustering_model.n_clusters
            self.means = torch.Tensor(
                clustering_model.cluster_centers_.reshape(self.num_clusters, 12, 2)
            ).cuda()
            self.normalize_direction = normalize_direction
            self.vars = vars
            self.active_cluster: Optional[int] = None
            self.requested_n_samples: int = 0
            self.get_sampleNum(20)

            #self.vars = vars
            #self.active_cluster: Optional[int] = None
        else:
            raise NotImplementedError

    def get_sampleNum(self, n_samples):
        if self.active_cluster is not None:
            self.sample_nums = np.zeros(self.num_clusters, dtype=int)
            self.sample_nums[self.active_cluster] = n_samples
            return

        _, label_counts = np.unique(self.clustering_model.labels_, return_counts=True)
        weights = label_counts / np.sum(label_counts)
        self.sample_nums = np.round(self.requested_n_samples * weights, 0).astype(int)

        while np.sum(self.sample_nums) != self.requested_n_samples:
            decimal = self.requested_n_samples * weights - self.sample_nums
            range_index = np.cumsum(self.sample_nums)
            if np.sum(self.sample_nums) > self.requested_n_samples:
                index = np.argmin(decimal)
                index_group = np.where(index < range_index)[0][0]
                self.sample_nums[index_group] -= 1
            else:
                index = np.argmax(decimal)
                index_group = np.where(index < range_index)[0][0]
                self.sample_nums[index_group] += 1

    def set_dist(self, base_pos):
        batch_size = base_pos.shape[0]
        self.dist = [None] * self.num_clusters
        clusters_to_use = (
            [self.active_cluster]
            if self.active_cluster is not None
            else list(range(self.num_clusters))
        )
        if not self.normalize_direction:
            direction = base_pos
            angles = torch.arctan2(direction[:, 1], direction[:, 0])
            rotate_matrix = (
                torch.stack(
                    [
                        torch.cos(angles),
                        torch.sin(angles),
                        -torch.sin(angles),
                        torch.cos(angles),
                    ]
                )
                .reshape(2, 2, -1)
                .permute([2, 0, 1])
            )  # (B,2,2)
            project_matrix = torch.abs(rotate_matrix)

            means_rotate = torch.matmul(
                self.means.unsqueeze(1), rotate_matrix.unsqueeze(0)
            )
            new_means_rotate = torch.cat(
                (
                    direction.unsqueeze(0)
                    .unsqueeze(-2)
                    .expand(self.num_clusters, -1, -1, -1),
                    means_rotate[:, :, :-1, :],
                ),
                dim=-2,
            )
            # new_means_rotate = means_rotate     # allfut
            vars_rotate = torch.matmul(
                self.vars.unsqueeze(1), project_matrix.unsqueeze(0)
            )

            for i in clusters_to_use:
                self.dist_i = Normal(
                    new_means_rotate[i], torch.clamp(vars_rotate[i], min=1e-4)
                )
                self.dist[i] = self.dist_i
        else:
            for i in clusters_to_use:
                self.dist_i = Normal(
                    self.means[i].unsqueeze(0).repeat(batch_size, 1, 1),
                    torch.clamp(
                        self.vars[i].unsqueeze(0).repeat(batch_size, 1, 1), min=1e-4
                    ),
                )
                self.dist[i] = self.dist_i

        # print(new_means_rotate.shape)     # (8,1024,12,2)
        # print(vars_rotate.shape)          # (8,1024,2)
        # print(self.vars.shape)        # (8,12,2)

        # construct dist

    def sample(self, n_sample=20):
        if np.sum(self.sample_nums) != n_sample:
            self.get_sampleNum(n_sample)
        samples = []
        cluster_indices = (
            [self.active_cluster]
            if self.active_cluster is not None
            else list(range(self.num_clusters))
        )
        for i in cluster_indices:
            d = self.dist[i]
            if d is None:
                continue
            samples_i = d.sample((n_sample,))  # (20,B,12,2)
            samples.append(samples_i[: self.sample_nums[i]])

        if not samples:
            raise ValueError("No distributions available for sampling.")

        samples = torch.cat(samples, dim=0).permute(1, 0, 2, 3)  # (B,20,12,2)
        return samples

    def sample_mean(self, sample_num=20):
        samples = []
        cluster_indices = (
            [self.active_cluster]
            if self.active_cluster is not None
            else list(range(self.num_clusters))
        )
        for i in cluster_indices:
            d = self.dist[i]
            if d is None:
                continue
            samples_i = d.loc.unsqueeze(0).expand([sample_num, -1, -1, -1])
            samples.append(samples_i[: self.sample_nums[i], :, 0, :])

        if not samples:
            raise ValueError("No distributions available for sampling.")

        samples = torch.cat(samples, dim=0).permute(1, 0, 2)
        return samples

    def log_prob(self, base_pos, x, u):
        if not self.normalize_direction:
            ## rotate
            direction = base_pos
            angles = -torch.arctan2(direction[:, 1], direction[:, 0])
            rotate_matrix = (
                torch.stack(
                    [
                        torch.cos(angles),
                        torch.sin(angles),
                        -torch.sin(angles),
                        torch.cos(angles),
                    ]
                )
                .reshape(2, 2, -1)
                .permute([2, 0, 1])
            )
            x_rotate = torch.matmul(x, rotate_matrix)

            clusters = self.clustering_model.predict(
                x_rotate.detach().reshape(-1, 24).cpu().numpy()
            )
        else:
            clusters = self.clustering_model.predict(
                x.detach().reshape(-1, 24).cpu().numpy()
            )
        ## cluster

        # probs = []
        # for d in self.dist:
        #     prob_i = d.log_prob(x)
        #     probs.append(prob_i.unsqueeze(0))
        # probs = torch.cat(probs)
        # traj_log_probs = probs.sum(-1).sum(-1)
        # _, max_traj = traj_log_probs.max(0)

        # global gmm_fit_global, count
        # gmm_fit = np.array([max_traj.cpu().numpy(), clusters])
        # gmm_fit_global.append(gmm_fit)
        # count += 1
        # if count == 10:
        #       print("10")

        ## compute
        log_prob_ = torch.zeros_like(u)
        cluster_indices = (
            [self.active_cluster]
            if self.active_cluster is not None
            else list(range(self.num_clusters))
        )
        for i in cluster_indices:
            dist = self.dist[i]
            if dist is None:
                continue
            mask = clusters == i
            if mask.any():
                log_prob_[mask] = dist.log_prob(u)[mask]

        return log_prob_

    def set_active_cluster(self, cluster_id: Optional[int]) -> None:
        if cluster_id is not None and (
            cluster_id < 0 or cluster_id >= self.num_clusters
        ):
            raise ValueError(
                f"cluster_id {cluster_id} outside of range [0, {self.num_clusters})"
            )
        self.active_cluster = cluster_id

    def override_cluster_stats(
        self,
        cluster_id: Optional[int],
        *,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> None:
        if cluster_id is None:
            return
        if mean is not None:
            mean_tensor = mean.to(self.means.device).view_as(self.means[cluster_id])
            self.means[cluster_id] = mean_tensor
        if std is not None:
            std_tensor = std.to(self.vars.device).view_as(self.vars[cluster_id])
            with torch.no_grad():
                self.vars.data[cluster_id] = std_tensor
