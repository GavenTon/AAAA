import argparse
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm, trange
from yacs.config import CfgNode

import wandb
from data.unified_loader import unified_loader
from metrics.build_metrics import Build_Metrics
from models.build_model import Build_Model
from utils.common import load_config, optimizer_to_cuda, set_seeds, set_wandb
from utils.finetune import (
    capture_reference_state,
    cluster_model_path,
    format_cluster_name,
    freeze_parameters_by_prefix,
    gather_cli,
    get_git_hash,
    load_prior_statistics,
    trim_optimizer,
)



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

    parser.add_argument("--finetune_from", type=str, default=None, help="checkpoint used as initialization")
    parser.add_argument("--cluster_id", type=int, default=None, help="target cluster id for finetuning")
    parser.add_argument("--epochs_finetune", type=int, default=None, help="number of finetuning epochs")
    parser.add_argument("--lr_finetune", type=float, default=None, help="override learning rate during finetuning")
    parser.add_argument("--freeze_backbone", action="store_true", help="freeze encoder/backbone layers during finetuning")
    parser.add_argument(
        "--freeze_prefixes",
        nargs="+",
        default=None,
        help="parameter name prefixes to freeze when --freeze_backbone is enabled",
    )
    parser.add_argument(
        "--prior_mode",
        type=str,
        choices=["cluster", "mixture"],
        default=None,
        help="prior mode when finetuning",
    )
    parser.add_argument(
        "--prior_stats_path",
        type=str,
        default=None,
        help="path to Gaussian prior statistics (mu, Sigma)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="root directory to store finetuned checkpoints and logs",
    )
    parser.add_argument(
        "--l2sp_weight",
        type=float,
        default=None,
        help="L2-SP regularization weight during finetuning",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=None,
        help="patience for early stopping during finetuning",
    )
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

    return parser.parse_args()


def aggregate(dict_list: List[Dict]) -> Dict:
    if not dict_list:
        return {}

    if "nsample" in dict_list[0]:
        ret_dict = {
            k: np.sum([d[k] for d in dict_list], axis=0)
            / np.sum([d["nsample"] for d in dict_list])
            for k in dict_list[0].keys()
        }
    else:
        ret_dict = {
            k: np.mean([d[k] for d in dict_list], axis=0) for k in dict_list[0].keys()
        }

    return ret_dict


def evaluate_model(
    cfg: CfgNode, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader
):
    model.eval()
    metrics = Build_Metrics(cfg)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    update_timesteps = [1]

    run_times = {0: []}
    run_times.update({t: [] for t in update_timesteps})

    result_info = {}

    print("evaluating ADE/FDE metrics ...")
    with torch.no_grad():
        result_list = []
        val_loss_list = []
        for i, data_dict in enumerate(tqdm(data_loader, leave=False)):
            data_dict = {
                k: (
                    data_dict[k].cuda()
                    if isinstance(data_dict[k], torch.Tensor)
                    else data_dict[k]
                )
                for k in data_dict
            }

            val_loss = model.update(data_dict, bp=False)
            val_loss_list.append(val_loss["loss"])
            dict_list = model.predict(deepcopy(data_dict), return_prob=False)

            dict_list = metrics.denormalize(dict_list)
            result_list.append(deepcopy(metrics(dict_list)))

        d = aggregate(result_list)
        result_info.update({k: d[k] for k in d.keys() if d[k] != 0.0})

    np.set_printoptions(precision=4)
    print(result_info)
    val_loss_info = np.array(val_loss_list).mean()

    model.train()
    return result_info, val_loss_info

def train_cluster_finetune(args, cfg) -> None:
    cluster_id = cfg.FINETUNE.CLUSTER_ID
    if cluster_id is None:
        raise ValueError("--cluster_id is required for finetuning")
    if args.finetune_from is None:
        raise ValueError("--finetune_from must be provided for cluster finetuning")

    cluster_tag = format_cluster_name(cluster_id)
    scene_name = cfg.DATA.DATASET_NAME
    base_save_dir = Path(cfg.FINETUNE.SAVE_DIR)
    if base_save_dir.name.lower() == scene_name.lower():
        save_root = base_save_dir
    else:
        save_root = base_save_dir / scene_name
    save_root.mkdir(parents=True, exist_ok=True)
    log_dir = save_root / cluster_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_root / f"{cluster_tag}.ckpt"

    cfg.defrost()
    cfg.OUTPUT_DIR = str(log_dir)
    cfg.freeze()

    cluster_model_file = cluster_model_path(cfg)

    data_loader = unified_loader(
        cfg,
        rand=True,
        split="train",
        batch_size=None,
        aug_scene=args.aug_scene,
        cluster_id=cluster_id,
        cluster_model_path=cluster_model_file,
    )

    validation = cfg.SOLVER.VALIDATION
    val_data_loader = None
    test_data_loader = None
    if validation:
        try:
            val_data_loader = unified_loader(
                cfg,
                rand=False,
                split="val",
                batch_size=None,
                aug_scene=args.aug_scene,
                cluster_id=cluster_id,
                cluster_model_path=cluster_model_file,
            )
        except ValueError as exc:
            print(f"Warning: {exc}. Skipping validation split during finetuning.")

        try:
            test_data_loader = unified_loader(
                cfg,
                rand=False,
                split="test",
                batch_size=None,
                aug_scene=args.aug_scene,
                cluster_id=cluster_id,
                cluster_model_path=cluster_model_file,
            )
        except ValueError as exc:
            print(f"Warning: {exc}. Skipping test split during finetuning.")

    has_val = validation and val_data_loader is not None
    has_test = validation and test_data_loader is not None

    model = Build_Model(cfg)
    model.load(Path(args.finetune_from), strict=False)

    freeze_summary = {"frozen": 0, "trainable": 0}
    if cfg.FINETUNE.FREEZE_BACKBONE:
        prefixes = cfg.FINETUNE.FREEZE_PREFIXES or ["encoder"]
        freeze_summary = freeze_parameters_by_prefix(model, prefixes)
        for optimizer in model.optimizers:
            trim_optimizer(optimizer)
    else:
        freeze_summary["trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    base_lr = cfg.SOLVER.LR
    finetune_lr = (
        cfg.FINETUNE.LR
        if cfg.FINETUNE.LR is not None
        else base_lr * cfg.FINETUNE.LR_SCALE
    )
    for optimizer in model.optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = finetune_lr

    prior_mean = prior_std = None
    if cfg.FINETUNE.PRIOR_MODE == "cluster":
        if cfg.FINETUNE.PRIOR_STATS_PATH:
            try:
                prior_mean, prior_std = load_prior_statistics(
                    Path(cfg.FINETUNE.PRIOR_STATS_PATH), cluster_id
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: failed to load prior statistics: {exc}")
        model.set_cluster_prior(cluster_id, mean=prior_mean, std=prior_std)
    else:
        model.set_active_cluster(None)

    if cfg.FINETUNE.L2SP_WEIGHT > 0:
        reference_state = capture_reference_state(model)
        model.set_regularizer(reference_state, cfg.FINETUNE.L2SP_WEIGHT)
    else:
        model.clear_regularizer()

    epochs = cfg.FINETUNE.EPOCHS
    patience = cfg.FINETUNE.EARLY_STOP_PATIENCE if cfg.FINETUNE.EARLY_STOP else None
    patience_counter = 0

    log_path = log_dir / "metrics.log"
    with log_path.open("w") as log_file:
        log_file.write(
            "epoch,train_loss,train_nll,reg_loss,val_ade,val_fde,test_ade,test_fde\n"
        )

    best_epoch = -1
    best_metrics = {"val_ade": float("inf"), "val_fde": float("inf")}
    best_train_loss = float("inf")
    best_state = None
    best_optimizer_state = None

    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            loss_list = []
            for data_dict in data_loader:
                data_dict = {
                    k: (
                        data_dict[k].cuda()
                        if isinstance(data_dict[k], torch.Tensor)
                        else data_dict[k]
                    )
                    for k in data_dict
                }

                if cfg.MGF.W_MSE == 0:
                    loss_list.append(model.update(data_dict))
                else:
                    loss_list.append(model.update_mse(data_dict, cfg.MGF.W_MSE))

            loss_info = aggregate(loss_list)
            pbar.set_postfix(OrderedDict(loss_info))

            val_ade_value = val_fde_value = None
            test_ade_value = test_fde_value = None
            val_flow_loss = test_flow_loss = None
            if has_val:
                val_results, val_flow_loss = evaluate_model(
                    cfg, model, val_data_loader
                )
                val_ade_value = val_results.get("ade")
                val_fde_value = val_results.get("fde")

            if has_test:
                test_results, test_flow_loss = evaluate_model(
                    cfg, model, test_data_loader
                )
                test_ade_value = test_results.get("ade")
                test_fde_value = test_results.get("fde")

            val_ade_log = (
                val_ade_value if val_ade_value is not None else float("nan")
            )
            val_fde_log = (
                val_fde_value if val_fde_value is not None else float("nan")
            )
            test_ade_log = (
                test_ade_value if test_ade_value is not None else float("nan")
            )
            test_fde_log = (
                test_fde_value if test_fde_value is not None else float("nan")
            )

            record = {
                "epoch": epoch,
                "train_loss": loss_info.get("loss", 0.0),
                "train_nll": loss_info.get("nll", loss_info.get("loss", 0.0)),
                "reg_loss": loss_info.get("reg_loss", 0.0),
                "val_ade": val_ade_log,
                "val_fde": val_fde_log,
                "test_ade": test_ade_log,
                "test_fde": test_fde_log,
            }

            with log_path.open("a") as log_file:
                log_file.write(
                    ",".join(
                        str(
                            record[key]
                            if not isinstance(record[key], float)
                            else f"{record[key]:.6f}"
                        )
                        for key in [
                            "epoch",
                            "train_loss",
                            "train_nll",
                            "reg_loss",
                            "val_ade",
                            "val_fde",
                            "test_ade",
                            "test_fde",
                        ]
                    )
                    + "\n"
                )

            if has_val and val_ade_value is not None and val_fde_value is not None:
                is_better = (
                    val_ade_value < best_metrics["val_ade"]
                    or val_fde_value < best_metrics["val_fde"]
                )
            else:
                is_better = record["train_loss"] < best_train_loss

            if is_better:
                if has_val and val_ade_value is not None and val_fde_value is not None:
                    best_metrics["val_ade"] = val_ade_value
                    best_metrics["val_fde"] = val_fde_value
                best_train_loss = record["train_loss"]
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                best_optimizer_state = deepcopy(model.optimizer.state_dict())
                patience_counter = 0
            elif patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": record["train_loss"],
                    "train_nll": record["train_nll"],
                    "reg_loss": record["reg_loss"],
                    "val_ade": val_ade_log,
                    "val_fde": val_fde_log,
                    "test_ade": test_ade_log,
                    "test_fde": test_fde_log,
                    "val_flow_loss": val_flow_loss,
                    "test_flow_loss": test_flow_loss,
                }
            )

    if best_state is None:
        best_state = deepcopy(model.state_dict())
        best_optimizer_state = deepcopy(model.optimizer.state_dict())
        best_epoch = epochs - 1

    model.load_state_dict(best_state)
    if best_optimizer_state is not None:
        model.optimizer.load_state_dict(best_optimizer_state)
        optimizer_to_cuda(model.optimizer)

    config_snapshot_path = log_dir / "config.yaml"
    with config_snapshot_path.open("w") as cfg_file:
        cfg_file.write(cfg.dump())

    args_snapshot_path = log_dir / "cli.json"
    with args_snapshot_path.open("w") as args_file:
        import json

        json.dump(gather_cli(args), args_file, indent=2)

    try:
        log_rel = str(log_path.relative_to(Path.cwd()))
    except ValueError:
        log_rel = str(log_path)
    try:
        cfg_rel = str(config_snapshot_path.relative_to(Path.cwd()))
    except ValueError:
        cfg_rel = str(config_snapshot_path)
    try:
        cli_rel = str(args_snapshot_path.relative_to(Path.cwd()))
    except ValueError:
        cli_rel = str(args_snapshot_path)

    metadata = {
        "cluster_id": cluster_id,
        "scene": scene_name,
        "best_epoch": best_epoch,
        "best_val_ade": best_metrics["val_ade"] if has_val else None,
        "best_val_fde": best_metrics["val_fde"] if has_val else None,
        "freeze_summary": freeze_summary,
        "finetune_lr": finetune_lr,
        "git_hash": get_git_hash(),
        "log_path": log_rel,
        "config_snapshot": cfg_rel,
        "cli_snapshot": cli_rel,
    }

    model.save(epoch=best_epoch, path=ckpt_path, extra=metadata)


def train(args, cfg) -> None:
    logging_path = cfg.OUTPUT_DIR
    validation = cfg.SOLVER.VALIDATION

    if hasattr(cfg, "FINETUNE") and cfg.FINETUNE.ENABLED:
        train_cluster_finetune(args, cfg)
        return

    data_loader = unified_loader(cfg, rand=True, split="train")
    if validation:
        val_data_loader = unified_loader(cfg, rand=False, split="val")
        val_loss = np.inf
        val_best_ade = np.inf
        val_best_fde = np.inf

        test_data_loader = unified_loader(cfg, rand=False, split="test")
        test_loss = np.inf
        test_best_ade = np.inf
        test_best_fde = np.inf

    start_epoch = 0
    model = Build_Model(cfg)

    if args.load_model is not None:
        # model saved at the end of each epoch. resume training from next epoch
        start_epoch = model.load(args.load_model) + 1
        # for optimizer in model.optimizers:
        #     for param_group in optimizer.param_groups:
        #         param_group['capturable'] = True
        print("loaded pretrained model")

    if cfg.SOLVER.USE_SCHEDULER:
        schedulers = [
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(cfg.SOLVER.ITER / 10),
                last_epoch=start_epoch - 1,
                gamma=0.7,
            )
            for optimizer in model.optimizers
        ]

    with tqdm(range(start_epoch, cfg.SOLVER.ITER)) as pbar:
        for i in pbar:
            loss_list = []
            for data_dict in data_loader:
                data_dict = {
                    k: (
                        data_dict[k].cuda()
                        if isinstance(data_dict[k], torch.Tensor)
                        else data_dict[k]
                    )
                    for k in data_dict
                }

                if cfg.MGF.W_MSE == 0:
                    loss_list.append(model.update(data_dict))
                else:
                    loss = model.update_mse(data_dict, cfg.MGF.W_MSE)
                    loss_list.append(loss)

            loss_info = aggregate(loss_list)
            pbar.set_postfix(OrderedDict(loss_info))

            # validation
            if (i + 1) % cfg.SOLVER.SAVE_EVERY == 0:
                if validation:
                    # val
                    val_results, val_flow_loss = evaluate_model(
                        cfg, model, val_data_loader
                    )
                    val_ade = val_results["ade"]
                    val_fde = val_results["fde"]

                    # test
                    test_results, test_flow_loss = evaluate_model(
                        cfg, model, test_data_loader
                    )
                    test_ade = test_results["ade"]
                    test_fde = test_results["fde"]

                    # save model based on val results
                    if val_ade < val_best_ade or val_fde < val_best_fde:
                        if val_ade < val_best_ade:
                            val_best_ade = val_ade
                            if args.save_model:
                                model.save(
                                    epoch=i, path=logging_path + f"/best_ade.ckpt"
                                )
                        if val_fde < val_best_fde:
                            val_best_fde = val_fde
                            if args.save_model:
                                model.save(
                                    epoch=i, path=logging_path + f"/best_fde.ckpt"
                                )

                    if (i + 1) % 25 == 0 and args.save_model:
                        model.save(
                            epoch=i,
                            path=logging_path
                            + f"/{i}_f{format(test_ade,'.3f')}_{format(test_fde,'.3f')}.ckpt",
                        )

            wandb.log(
                {
                    "epoch": i,
                    "train_loss": loss_info["loss"],
                    "val_flow_loss": val_flow_loss,
                    "test_flow_loss": test_flow_loss,
                    "val_ade": val_ade,
                    "val_fde": val_fde,
                    "test_ade": test_ade,
                    "test_fde": test_fde,
                    "val_best_ade": val_best_ade,
                    "val_best_fde": val_best_fde,
                }
            )
            if cfg.MGF.W_MSE != 0:
                wandb.log(
                    {
                        "epoch": i,
                        "train_flow_loss": loss_info["flow_loss"],
                        "train_msemin_loss": loss_info["mse_min_loss"],
                    }
                )

        if cfg.SOLVER.USE_SCHEDULER:
            [scheduler.step() for scheduler in schedulers]

    return


if __name__ == "__main__":
    args = parse_args()
    args.config_file = f"./config/{args.scene}.yml"
    cfg = load_config(args)
    cfg.defrost()

    finetune_requested = args.finetune_from is not None or args.cluster_id is not None
    if finetune_requested:
        if args.finetune_from is None or args.cluster_id is None:
            raise ValueError(
                "--finetune_from and --cluster_id must be provided together for finetuning"
            )
        cfg.FINETUNE.ENABLED = True
        cfg.FINETUNE.CLUSTER_ID = args.cluster_id
        if args.epochs_finetune is not None:
            cfg.FINETUNE.EPOCHS = args.epochs_finetune
        if args.lr_finetune is not None:
            cfg.FINETUNE.LR = args.lr_finetune
        if args.prior_mode is not None:
            cfg.FINETUNE.PRIOR_MODE = args.prior_mode
        if args.prior_stats_path is not None:
            cfg.FINETUNE.PRIOR_STATS_PATH = args.prior_stats_path
        if args.save_dir is not None:
            cfg.FINETUNE.SAVE_DIR = args.save_dir
        cfg.FINETUNE.FREEZE_BACKBONE = args.freeze_backbone
        if args.freeze_prefixes is not None:
            cfg.FINETUNE.FREEZE_PREFIXES = args.freeze_prefixes
        if args.l2sp_weight is not None:
            cfg.FINETUNE.L2SP_WEIGHT = args.l2sp_weight
        if args.early_stop_patience is not None:
            cfg.FINETUNE.EARLY_STOP = args.early_stop_patience > 0
            if args.early_stop_patience > 0:
                cfg.FINETUNE.EARLY_STOP_PATIENCE = args.early_stop_patience
    else:
        cfg.FINETUNE.ENABLED = False

    cfg.freeze()
    set_wandb(args)
    set_seeds(random.randint(0, 1000))

    train(args, cfg)
