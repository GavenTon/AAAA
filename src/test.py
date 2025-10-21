import argparse
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm, trange
from typing import Dict, List, Optional,Tuple
from torch.utils.data import DataLoader, Subset

from data.TP.preprocessing import dict_collate
from data.router_dataset import extract_router_features, ClusterRoutingDataset
from data.unified_loader import (
    load_cluster_model,
    load_environment_dataset,
    predict_cluster_from_dict,
    unified_loader,
)
from metrics.build_metrics import Build_Metrics
from models.build_model import Build_Model
from models.router import ClusterRouter
from utils.common import load_config, set_seeds
from utils.finetune import cluster_model_path, format_cluster_name

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch training & testing code for task-agnostic time-series prediction"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--mode", type=str, choices=["train", "test", "tune"], default="train"
    )

    parser.add_argument("--model_name", type=str)
    parser.add_argument("--save_model", action="store_true", help="save model")
    parser.add_argument(
        "--load_model", type=str, default=None, help="path of pre-trained model"
    )
    parser.add_argument("--logging_path", type=str, default=None)

    parser.add_argument(
        "--config_root",
        type=str,
        default="config/",
        help="root path to config file",
    )
    parser.add_argument("--scene", type=str, default="eth", help="scene name")

    parser.add_argument(
        "--aug_scene", action="store_true", help="trajectron++ augmentation"
    )
    parser.add_argument(
        "--w_mse", type=float, default=0, help="loss weight of mse_loss"
    )

    parser.add_argument("--clusterGMM", action="store_true")
    parser.add_argument(
        "--cluster_method", type=str, default="kmeans", help="clustering method"
    )
    parser.add_argument("--cluster_n", type=int, help="n cluster centers")
    parser.add_argument(
        "--cluster_name", type=str, default="", help="clustering model name"
    )
    parser.add_argument("--manual_weights", nargs="+", default=None, type=int)

    parser.add_argument("--var_init", type=float, default=0.7, help="init var")
    parser.add_argument("--learnVAR", action="store_true")

    parser.add_argument(
        "--load_cluster_ckpt",
        type=str,
        default=None,
        help="path to a cluster-specific checkpoint",
    )
    parser.add_argument(
        "--cluster_id",
        type=int,
        default=None,
        help="evaluate metrics for a single cluster",
    )
    parser.add_argument(
        "--router_ckpt",
        type=str,
        default=None,
        help="path to a trained cluster router checkpoint",
    )
    parser.add_argument(
        "--cluster_ckpt_dir",
        type=str,
        default=None,
        help="directory containing finetuned cluster checkpoints",
    )

    return parser.parse_args()


def k_means(batch_x, ncluster=20, iter=10):
    B, N, D = batch_x.size()
    batch_c = torch.Tensor().cuda()
    for i in trange(B):
        x = batch_x[i]
        c = x[torch.randperm(N)[:ncluster]]
        for i in range(iter):
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            c[nanix] = x[torch.randperm(N)[:ndead]]

        batch_c = torch.cat((batch_c, c.unsqueeze(0)), dim=0)
    return batch_c


def run_inference(cfg, model, metrics, data_loader):
    model.eval()
    with torch.no_grad():
        pred_list = []
        gt_list = []
        obs_list = []

        for i, data_dict in enumerate(tqdm(data_loader, leave=False)):
            pred_list_i = []
            gt_list_i = []

            data_dict = {
                k: (
                    data_dict[k].cuda()
                    if isinstance(data_dict[k], torch.Tensor)
                    else data_dict[k]
                )
                for k in data_dict
            }

            dist_args = model.encoder(data_dict)

            if cfg.MGF.ENABLE:
                base_pos = model.get_base_pos(data_dict).clone()
            else:
                base_pos = (
                    model.get_base_pos(data_dict)[:, None]
                    .expand(-1, cfg.MGF.POST_CLUSTER, -1)
                    .clone()
                )  # (B, 20, 2)
            dist_args = dist_args[:, None].expand(-1, cfg.MGF.POST_CLUSTER, -1, -1)

            sampled_seq = model.flow.sample(
                base_pos, cond=dist_args, n_sample=cfg.MGF.POST_CLUSTER
            )

            dict_list = []
            for i in range(cfg.MGF.POST_CLUSTER):
                data_dict_i = deepcopy(data_dict)
                data_dict_i[("pred_st", 0)] = sampled_seq[:, i]
                if torch.sum(torch.isnan(data_dict_i[("pred_st", 0)])):
                    data_dict_i[("pred_st", 0)] = torch.where(
                        torch.isnan(data_dict_i[("pred_st", 0)]),
                        data_dict_i["obs_st"][:, 0, None, 2:4].expand(
                            data_dict_i[("pred_st", 0)].size()
                        ),
                        data_dict_i[("pred_st", 0)],
                    )
                dict_list.append(data_dict_i)

            dict_list = metrics.denormalize(dict_list)

            for data_dict in dict_list:
                obs_list_i = data_dict["obs"].cpu().numpy()
                pred_traj_i = data_dict[("pred", 0)].cpu().numpy()  # (B,12,2)
                pred_list_i.append(pred_traj_i)
                gt_list_i = data_dict["gt"].cpu().numpy()
            pred_list_i = np.array(pred_list_i).transpose(1, 0, 2, 3)

            pred_list.append(pred_list_i)
            gt_list.append(gt_list_i)
            obs_list.append(obs_list_i)

    pred_list = np.concatenate(pred_list, axis=0)
    gt_list = np.concatenate(gt_list, axis=0)
    obs_list = np.concatenate(obs_list, axis=0)

    pred_list_flatten = torch.Tensor(
        pred_list.reshape(pred_list.shape[0], cfg.MGF.POST_CLUSTER, -1)
    ).cuda()
    pred_list = (
        k_means(pred_list_flatten).cpu().numpy().reshape(pred_list.shape[0], 20, -1, 2)
    )

    return obs_list, gt_list, pred_list


def evaluate_metrics(args, gt_list, pred_list):
    l2_dis = np.sqrt(((pred_list - gt_list[:, np.newaxis, :, :]) ** 2).sum(-1))
    minade = l2_dis.mean(-1).min(-1)
    minfde = l2_dis[:, :, -1].min(-1)
    if args.scene == "eth":
        minade /= 0.6
        minfde /= 0.6
    elif args.scene == "sdd":
        minade *= 50
        minfde *= 50
    return minade.mean(), minfde.mean()

def _evaluate_router_assignments(
    args,
    cfg,
    router_model: ClusterRouter,
    feature_set: str,
    flatten: bool,
    normalization_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    print(
        f"[DEBUG] Starting router assignment evaluation | feature_set={feature_set} | flatten={flatten}"
    )
    device = next(router_model.parameters()).device
    if normalization_stats is not None:
        mean, std = normalization_stats
        normalization_stats = (mean.to(device), std.to(device))
        print(
            "[DEBUG] Normalization stats available | mean[:5]={:.4f}, std[:5]={:.4f}".format(
                normalization_stats[0].flatten()[:5].mean().item(),
                normalization_stats[1].flatten()[:5].mean().item(),
            )
        )
    else:
        print("[DEBUG] No normalization stats provided; using raw features")
    dataset = load_environment_dataset(cfg, split="test", aug_scene=args.aug_scene)
    print(f"[DEBUG] Loaded dataset split=test with {len(dataset)} samples")
    cluster_model = load_cluster_model(cfg, cluster_model_path(cfg))
    assignments = {cid: [] for cid in range(router_model.num_clusters)}
    correct = 0
    total = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample is None:
                continue
            data_dict = dict_collate([sample])
            features = extract_router_features(
                data_dict, feature_set=feature_set, flatten=flatten
            ).to(device)
            if normalization_stats is not None:
                mean, std = normalization_stats
                std_safe = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
                features = (features - mean) / std_safe
            pred_cluster = int(router_model.predict(features)[0].item())
            assignments[pred_cluster].append(idx)
            true_cluster = predict_cluster_from_dict(
                data_dict,
                cluster_model=cluster_model,
                normalize_direction=cfg.DATA.NORMALIZED,
            )
            if true_cluster is not None:
                total += 1
                if true_cluster == pred_cluster:
                    correct += 1
            if idx < 5:
                print(
                    "[DEBUG] Sample {} | pred_cluster={} | true_cluster={}".format(
                        idx, pred_cluster, true_cluster
                    )
                )

    accuracy = correct / total if total else 0.0
    print(
        "[DEBUG] Completed router assignment pass | total_evaluated={} | correct={} | accuracy={:.4f}".format(
            total, correct, accuracy
        )
    )
    print(
        "[DEBUG] Routed sample counts by cluster: {}".format(
            {cid: len(idxs) for cid, idxs in assignments.items()}
        )
    )
    return assignments, accuracy, dataset


def evaluate_with_router(args, cfg):
    if args.router_ckpt is None:
        raise ValueError("--router_ckpt must be provided for router-based evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router_model, checkpoint = ClusterRouter.from_checkpoint(args.router_ckpt, device=device)
    router_model.eval()
    print(
        "[DEBUG] Loaded router checkpoint {} | mode={}".format(
            args.router_ckpt, "eval" if not router_model.training else "train"
        )
    )
    metadata = checkpoint.get("metadata", {})
    feature_set = metadata.get("feature_set", cfg.ROUTER.FEATURE_SET)
    flatten = metadata.get("flatten", cfg.ROUTER.FLATTEN)
    norm_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    feature_mean = metadata.get("feature_mean")
    feature_std = metadata.get("feature_std")
    if feature_mean is not None and feature_std is not None:
        mean_tensor = torch.as_tensor(feature_mean, dtype=torch.float32, device=device)
        std_tensor = torch.as_tensor(feature_std, dtype=torch.float32, device=device)
        norm_stats = (mean_tensor, std_tensor)
    else:
        try:
            print("[DEBUG] Router checkpoint missing normalization stats; recomputing from training split")
            stats_dataset = ClusterRoutingDataset(
                cfg,
                split="train",
                feature_set=feature_set,
                flatten=flatten,
                cluster_model_path=cluster_model_path(cfg),
                aug_scene=args.aug_scene,
            )
            norm_stats = (
                stats_dataset.feature_mean.to(device),
                stats_dataset.feature_std.to(device),
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(
                f"Warning: unable to recompute router normalization statistics ({exc}). Proceeding without normalization."
            )

    assignments, accuracy, dataset = _evaluate_router_assignments(
        args, cfg, router_model, feature_set, flatten, normalization_stats=norm_stats
    )

    save_root = (
        Path(args.cluster_ckpt_dir)
        if args.cluster_ckpt_dir
        else Path(cfg.FINETUNE.SAVE_DIR) / cfg.DATA.DATASET_NAME
    )

    aggregated_gt = []
    aggregated_pred = []
    per_cluster_metrics = {}

    for cluster_id, indices in assignments.items():
        if not indices:
            continue
        ckpt_path = save_root / f"{format_cluster_name(cluster_id)}.ckpt"
        if not ckpt_path.exists():
            print(
                f"Warning: cluster checkpoint not found for cluster {cluster_id} at {ckpt_path}. Skipping."
            )
            continue

        model = Build_Model(cfg)
        model.load(ckpt_path, strict=False)
        model.set_active_cluster(cluster_id)

        subset = Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=cfg.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATA.NUM_WORKERS,
            collate_fn=dict_collate,
            drop_last=False,
            pin_memory=True,
        )

        metrics = Build_Metrics(cfg)
        obs, gt, pred = run_inference(cfg, model, metrics, loader)
        aggregated_gt.append(gt)
        aggregated_pred.append(pred)

        minade, minfde = evaluate_metrics(args, gt, pred)
        per_cluster_metrics[cluster_id] = {"minADE": minade, "minFDE": minfde}

    if not aggregated_gt:
        raise RuntimeError("Router did not route any samples to available clusters")

    gt_all = np.concatenate(aggregated_gt, axis=0)
    pred_all = np.concatenate(aggregated_pred, axis=0)
    minade, minfde = evaluate_metrics(args, gt_all, pred_all)

    print(f"Router accuracy on clustering labels: {accuracy:.4f}")
    for cluster_id, metrics_dict in sorted(per_cluster_metrics.items()):
        print(
            f"Cluster {cluster_id}: minADE={metrics_dict['minADE']:.4f}, minFDE={metrics_dict['minFDE']:.4f}"
        )
    print(f"Aggregated test metrics -> minADE: {minade:.4f}, minFDE: {minfde:.4f}")


def test(args, cfg):
    if args.router_ckpt:
        evaluate_with_router(args, cfg)
        return
    model = Build_Model(cfg)
    load_path = Path(args.load_cluster_ckpt) if args.load_cluster_ckpt else Path(args.load_model)
    model.load(load_path, strict=False)
    if args.cluster_id is not None:
        model.set_active_cluster(args.cluster_id)
        data_loader = unified_loader(
            cfg,
            rand=False,
            split="test",
            cluster_id=args.cluster_id,
            cluster_model_path=cluster_model_path(cfg),
        )
    else:
        data_loader = unified_loader(cfg, rand=False, split="test")
    metrics = Build_Metrics(cfg)

    for i_trial in range(3):
        set_seeds(random.randint(0, 1000))
        _, gt_list, pred_list = run_inference(cfg, model, metrics, data_loader)
        minade, minfde = evaluate_metrics(args, gt_list, pred_list)
        print(f"{args.scene} test {i_trial}:\n {minade}/{minfde}")


if __name__ == "__main__":
    args = parse_args()
    scene = args.scene
    if args.load_cluster_ckpt is None:
        args.load_model = f"./checkpoint/{scene}.ckpt"
    args.config_file = f"./config/{scene}.yml"
    cfg = load_config(args)
    cfg.freeze()

    test(args, cfg)
