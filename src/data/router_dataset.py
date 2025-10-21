from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode

from data.TP.preprocessing import dict_collate
from data.unified_loader import (
    load_cluster_model,
    load_environment_dataset,
    predict_cluster_from_dict,
)


def _infer_num_clusters(cluster_model) -> int:
    if hasattr(cluster_model, "n_clusters"):
        return int(cluster_model.n_clusters)
    if hasattr(cluster_model, "cluster_centers_"):
        return int(len(cluster_model.cluster_centers_))
    raise AttributeError("Unable to infer cluster count from clustering model")


class ClusterRoutingDataset(Dataset):
    """Dataset that exposes per-sample features and cluster labels for router training."""

    def __init__(
        self,
        cfg: CfgNode,
        split: str = "train",
        feature_set: str = "obs_st",
        flatten: bool = True,
        cluster_model_path: Optional[Path] = None,
        aug_scene: bool = False,
        normalization_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        if cfg.DATA.TASK != "TP":
            raise NotImplementedError("Cluster router supports only TP task datasets.")

        self.cfg = cfg
        self.split = split
        self.feature_set = feature_set
        self.flatten = flatten
        self.dataset = load_environment_dataset(cfg, split=split, aug_scene=aug_scene)
        self.cluster_model = load_cluster_model(cfg, cluster_model_path)
        self.num_clusters = _infer_num_clusters(self.cluster_model)

        indices: List[int] = []
        labels: List[int] = []
        feature_bank: List[torch.Tensor] = []
        self._skipped_nan: int = 0

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if sample is None:
                continue
            data_dict = dict_collate([sample])
            feature = extract_router_features(
                data_dict,
                feature_set=self.feature_set,
                flatten=self.flatten,
            )
            feature = torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
            if not torch.isfinite(feature).all():
                self._skipped_nan += 1
                continue
            cluster = predict_cluster_from_dict(
                data_dict,
                cluster_model=self.cluster_model,
                normalize_direction=cfg.DATA.NORMALIZED,
            )
            if cluster is None:
                continue
            feature_bank.append(feature.float())
            indices.append(idx)
            labels.append(int(cluster))

        if not labels:
            raise ValueError(
                f"No usable samples found for split '{split}' while building router dataset."
            )

        raw_features = torch.stack(feature_bank, dim=0)
        raw_features = raw_features.to(torch.float32)

        if normalization_stats is None:
            feature_mean = raw_features.mean(dim=0)
            feature_std = raw_features.std(dim=0)
        else:
            feature_mean, feature_std = normalization_stats

        feature_mean = feature_mean.clone().to(raw_features.dtype)
        feature_std = feature_std.clone().to(raw_features.dtype)
        feature_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)

        normalized = (raw_features - feature_mean) / feature_std
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        self._features = normalized
        self._labels = torch.tensor(labels, dtype=torch.long)
        self._indices = indices
        self.feature_dim = normalized.shape[1]
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        class_counts = torch.bincount(self._labels, minlength=self.num_clusters)
        self.class_counts = class_counts
        valid_mask = class_counts > 0
        class_weights = torch.zeros_like(class_counts, dtype=torch.float32)
        if valid_mask.any():
            denom = class_counts[valid_mask].float()
            class_weights[valid_mask] = class_counts.sum().float() / (
                denom * valid_mask.sum().float()
            )
        else:
            class_weights += 1.0
        self.class_weights = class_weights
        self.sample_weights = class_weights[self._labels].double()

        missing_clusters = (class_counts == 0).nonzero(as_tuple=False).flatten()
        if missing_clusters.numel() > 0:
            print(
                f"Warning: router dataset split '{split}' missing clusters: {missing_clusters.tolist()}"
            )

        if self._skipped_nan > 0:
            print(
                f"Warning: skipped {self._skipped_nan} samples with non-finite router features in split '{split}'."
            )

    def __len__(self) -> int:
        return self._labels.numel()

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        feature = self._features[item]
        label = int(self._labels[item])
        return feature, label

    def subset_indices(self, indices: List[int]) -> List[int]:
        return [self._indices[i] for i in indices]


def extract_router_features(
    data_dict: dict,
    feature_set: str = "obs_st",
    flatten: bool = True,
) -> torch.Tensor:
    if feature_set == "obs_st":
        base = data_dict["obs_st"].squeeze(0).float()
    elif feature_set == "obs":
        base = data_dict["obs"].squeeze(0).float()
    else:
        raise ValueError(f"Unsupported feature set '{feature_set}' for router training")

    base = torch.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if base.dim() > 2:
        base = base.reshape(base.shape[0], -1)

    components = [base]
    if base.shape[0] > 1:
        vel = torch.zeros_like(base)
        vel[1:] = base[1:] - base[:-1]
        components.append(vel)
    else:
        vel = None
    if vel is not None and base.shape[0] > 2:
        acc = torch.zeros_like(base)
        acc[2:] = vel[2:] - vel[1:-1]
        components.append(acc)

    feature = torch.cat(components, dim=-1)

    if flatten:
        feature = feature.reshape(-1)

    return feature.float()
