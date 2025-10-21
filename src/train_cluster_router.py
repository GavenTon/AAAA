from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.router_dataset import ClusterRoutingDataset
from models.router import ClusterRouter, RouterConfig
from utils.common import load_config, set_seeds
from utils.finetune import gather_cli, get_git_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a high-level cluster router")
    parser.add_argument("--scene", type=str, default="eth", help="target scene name")
    parser.add_argument(
        "--config_root", type=str, default="config", help="directory with scene configs"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="explicit path to config file (overrides --scene)",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature_set",
        type=str,
        choices=["obs_st", "obs"],
        default=None,
        help="feature set used for routing",
    )
    parser.add_argument("--cluster_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="directory where router checkpoints and logs will be stored",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="number of epochs without improvement before stopping",
    )
    parser.add_argument(
        "--no_eval_test",
        action="store_true",
        help="skip test split evaluation after training",
    )
    return parser.parse_args()


def _prepare_cfg(args: argparse.Namespace):
    args.mode = "train"
    args.model_name = "cluster_router"
    if not args.config_file:
        args.config_file = str(Path(args.config_root) / f"{args.scene}.yml")
    cfg = load_config(args)
    cfg.defrost()
    if args.feature_set is not None:
        cfg.ROUTER.FEATURE_SET = args.feature_set
    if args.batch_size is not None:
        cfg.ROUTER.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        cfg.ROUTER.EPOCHS = args.epochs
    if args.lr is not None:
        cfg.ROUTER.LR = args.lr
    if args.weight_decay is not None:
        cfg.ROUTER.WEIGHT_DECAY = args.weight_decay
    if args.hidden_dim is not None:
        cfg.ROUTER.HIDDEN_DIM = args.hidden_dim
    if args.num_layers is not None:
        cfg.ROUTER.NUM_LAYERS = args.num_layers
    if args.dropout is not None:
        cfg.ROUTER.DROPOUT = args.dropout
    if args.early_stop_patience is not None:
        cfg.ROUTER.EARLY_STOP_PATIENCE = args.early_stop_patience
    if args.save_dir is not None:
        cfg.ROUTER.SAVE_DIR = args.save_dir
    cfg.freeze()
    return cfg


def _build_dataloader(
    dataset: ClusterRoutingDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def _run_epoch(
    model: ClusterRouter,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(mode=training)
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    for features, labels in tqdm(loader, leave=False):
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        preds = torch.argmax(logits, dim=-1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    if total_samples == 0:
        return 0.0, 0.0
    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc


def _maybe_build_dataset(
    cfg,
    split: str,
    feature_set: str,
    cluster_model_path: Optional[str],
    normalization_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Optional[ClusterRoutingDataset]:
    try:
        return ClusterRoutingDataset(
            cfg,
            split=split,
            feature_set=feature_set,
            flatten=cfg.ROUTER.FLATTEN,
            cluster_model_path=Path(cluster_model_path) if cluster_model_path else None,
            normalization_stats=normalization_stats,
        )
    except (FileNotFoundError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    cfg = _prepare_cfg(args)

    train_dataset = _maybe_build_dataset(
        cfg,
        "train",
        cfg.ROUTER.FEATURE_SET,
        args.cluster_model_path,
    )
    if train_dataset is None:
        raise RuntimeError("Unable to build training dataset for router")

    val_dataset = _maybe_build_dataset(
        cfg,
        "val",
        cfg.ROUTER.FEATURE_SET,
        args.cluster_model_path,
        normalization_stats=(train_dataset.feature_mean, train_dataset.feature_std),
    )
    test_dataset = None
    if not args.no_eval_test:
        test_dataset = _maybe_build_dataset(
            cfg,
            "test",
            cfg.ROUTER.FEATURE_SET,
            args.cluster_model_path,
            normalization_stats=(train_dataset.feature_mean, train_dataset.feature_std),
        )

    sampler = None
    if cfg.ROUTER.BALANCE_CLASSES and train_dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            num_samples=len(train_dataset.sample_weights),
            replacement=True,
        )

    train_loader = _build_dataloader(
        train_dataset,
        cfg.ROUTER.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.DATA.NUM_WORKERS,
        sampler=sampler,
    )
    val_loader = (
        _build_dataloader(
            val_dataset,
            cfg.ROUTER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            sampler=None,
        )
        if val_dataset is not None
        else None
    )
    test_loader = (
        _build_dataloader(
            test_dataset,
            cfg.ROUTER.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            sampler=None,
        )
        if test_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router_config = RouterConfig(
        input_dim=train_dataset.feature_dim,
        num_clusters=train_dataset.num_clusters,
        hidden_dim=cfg.ROUTER.HIDDEN_DIM,
        num_layers=cfg.ROUTER.NUM_LAYERS,
        dropout=cfg.ROUTER.DROPOUT,
        use_input_layer_norm=cfg.ROUTER.INPUT_LAYER_NORM,
    )
    model = ClusterRouter(router_config).to(device)
    class_weights = None
    if cfg.ROUTER.BALANCE_CLASSES and train_dataset.class_weights is not None:
        class_weights = train_dataset.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.ROUTER.LR, weight_decay=cfg.ROUTER.WEIGHT_DECAY
    )

    best_metric = float("-inf")
    best_state: Dict[str, torch.Tensor] = {}
    patience = cfg.ROUTER.EARLY_STOP_PATIENCE
    epochs_no_improve = 0

    history = {"train": [], "val": []}

    for epoch in range(1, cfg.ROUTER.EPOCHS + 1):
        print(f"Epoch {epoch}/{cfg.ROUTER.EPOCHS}")
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        history["train"].append({"loss": train_loss, "acc": train_acc})

        val_metric = train_acc
        if val_loader is not None:
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, None, device)
            history["val"].append({"loss": val_loss, "acc": val_acc})
            val_metric = val_acc
            print(
                f"  train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}"
            )
        else:
            print(f"  train loss {train_loss:.4f} acc {train_acc:.4f}")

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    if best_state:
        model.load_state_dict(best_state)

    if test_loader is not None:
        test_loss, test_acc = _run_epoch(model, test_loader, criterion, None, device)
    else:
        test_loss, test_acc = 0.0, 0.0

    save_root = Path(cfg.ROUTER.SAVE_DIR) / cfg.DATA.DATASET_NAME
    save_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_root / "router.ckpt"
    metadata = {
        "config": cfg.dump(),
        "cli_args": gather_cli(args),
        "git_hash": get_git_hash(),
        "best_metric": best_metric,
        "feature_set": cfg.ROUTER.FEATURE_SET,
        "flatten": cfg.ROUTER.FLATTEN,
        "feature_mean": train_dataset.feature_mean.cpu().tolist(),
        "feature_std": train_dataset.feature_std.cpu().tolist(),
        "class_counts": train_dataset.class_counts.cpu().tolist(),
    }
    model.save_checkpoint(str(checkpoint_path), metadata=metadata, optimizer=optimizer)

    metrics_path = save_root / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "history": history,
                "best_metric": best_metric,
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
            f,
            indent=2,
        )

    print(f"Router checkpoint saved to {checkpoint_path}")
    if test_loader is not None:
        print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
