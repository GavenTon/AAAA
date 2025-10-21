from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from yacs.config import CfgNode


def format_cluster_name(cluster_id: int, width: int = 2) -> str:
    return f"cluster_{cluster_id:0{width}d}"


def cluster_model_path(cfg: CfgNode) -> Path:
    return (
        Path("src")
        / "clustering"
        / "models"
        / f"{cfg.DATA.DATASET_NAME}_train_{cfg.MGF.CLUSTER_METHOD}_{str(cfg.MGF.CLUSTER_N)}.pkl"
    )


def freeze_parameters_by_prefix(
    model: torch.nn.Module,
    prefixes: Sequence[str],
) -> Dict[str, int]:
    summary = {"frozen": 0, "trainable": 0}
    prefixes = tuple(prefixes)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if prefixes and name.startswith(prefixes):
            param.requires_grad = False
            summary["frozen"] += param.numel()
        else:
            summary["trainable"] += param.numel()
    return summary


def trim_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for group in list(optimizer.param_groups):
        params = [p for p in group["params"] if p.requires_grad]
        removed = [p for p in group["params"] if not p.requires_grad]
        group["params"] = params
        for param in removed:
            if param in optimizer.state:
                del optimizer.state[param]
    optimizer.param_groups = [g for g in optimizer.param_groups if g["params"]]


def capture_reference_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    reference: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            reference[name] = param.detach().clone()
    return reference


def load_prior_statistics(
    stats_path: Path,
    cluster_id: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not stats_path.exists():
        raise FileNotFoundError(f"prior statistics file not found: {stats_path}")

    data = np.load(stats_path)
    mu = data.get("mu")
    sigma = data.get("Sigma")
    if mu is None or sigma is None:
        raise KeyError("prior statistics must contain 'mu' and 'Sigma'")

    mu_cluster = np.asarray(mu)[cluster_id]
    sigma_cluster = np.asarray(sigma)[cluster_id]

    if mu_cluster.ndim == 1:
        mu_cluster = mu_cluster.reshape(-1, 2)
    if mu_cluster.ndim == 2 and mu_cluster.shape[-1] != 2:
        raise ValueError("mu entries must be shaped (T, 2)")

    if sigma_cluster.ndim == 1:
        sigma_cluster = sigma_cluster.reshape(-1, 2)
    elif sigma_cluster.ndim == 2 and sigma_cluster.shape[0] == mu_cluster.shape[0] and sigma_cluster.shape[1] != 2:
        sigma_cluster = np.sqrt(np.clip(np.diag(sigma_cluster), 1e-6, None)).reshape(-1, 2)
    elif sigma_cluster.ndim == 3:
        sigma_cluster = np.sqrt(
            np.clip(np.diagonal(sigma_cluster, axis1=-2, axis2=-1), 1e-6, None)
        )
    if sigma_cluster.shape != mu_cluster.shape:
        raise ValueError("mu and Sigma must share the same shape after processing")

    mu_tensor = torch.from_numpy(mu_cluster.astype(np.float32))
    sigma_tensor = torch.from_numpy(sigma_cluster.astype(np.float32))
    return mu_tensor, sigma_tensor


def gather_cli(args_namespace) -> Dict[str, object]:
    return json.loads(json.dumps(vars(args_namespace)))


def get_git_hash() -> str:
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd())
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"
