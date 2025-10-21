from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RouterConfig:
    input_dim: int
    num_clusters: int
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    use_input_layer_norm: bool = True


class ClusterRouter(nn.Module):
    def __init__(self, config: RouterConfig) -> None:
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("Router must contain at least one layer")

        layers = []
        self.input_norm = (
            nn.LayerNorm(config.input_dim) if config.use_input_layer_norm else None
        )
        in_dim = config.input_dim
        for layer_idx in range(config.num_layers - 1):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.num_clusters))
        self.network = nn.Sequential(*layers)

        self._input_dim = config.input_dim
        self._num_clusters = config.num_clusters
        self._hidden_dim = config.hidden_dim
        self._num_layers = config.num_layers
        self._dropout = config.dropout
        self._config = config

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if self.input_norm is not None:
            features = self.input_norm(features)
        return self.network(features)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features)
        return torch.argmax(logits, dim=-1)

    def save_checkpoint(
        self,
        path: str,
        metadata: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        package: Dict[str, Any] = {
            "model_state": self.state_dict(),
            "router_meta": {
                "input_dim": self._input_dim,
                "num_clusters": self._num_clusters,
                "hidden_dim": self._hidden_dim,
                "num_layers": self._num_layers,
                "dropout": self._dropout,
                "use_input_layer_norm": self._config.use_input_layer_norm,
            },
        }
        if metadata is not None:
            package["metadata"] = metadata
        if optimizer is not None:
            package["optim_state"] = optimizer.state_dict()
        torch.save(package, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location)
        state = checkpoint.get("model_state", checkpoint)
        self.load_state_dict(state)
        return checkpoint

    @classmethod
    def from_checkpoint(
        cls, path: str, device: Optional[torch.device] = None
    ) -> Tuple["ClusterRouter", Dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device if device else "cpu")
        meta = checkpoint.get("router_meta")
        if meta is None:
            raise KeyError("router_meta missing from checkpoint")
        config = RouterConfig(
            input_dim=meta["input_dim"],
            num_clusters=meta["num_clusters"],
            hidden_dim=meta.get("hidden_dim", 256),
            num_layers=meta.get("num_layers", 2),
            dropout=meta.get("dropout", 0.1),
            use_input_layer_norm=meta.get("use_input_layer_norm", True),
        )
        model = cls(config)
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
        if device:
            model.to(device)
        return model, checkpoint
