import argparse
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm, trange
from yacs.config import CfgNode

import wandb
from data.unified_loader import unified_loader
from metrics.build_metrics import Build_Metrics
from models.build_model import Build_Model
from utils.common import load_config, set_seeds, set_wandb


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

    return parser.parse_args()


def aggregate(dict_list: List[Dict]) -> Dict:
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


def train(args, cfg) -> None:
    logging_path = cfg.OUTPUT_DIR
    validation = cfg.SOLVER.VALIDATION

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
    cfg.freeze()
    set_wandb(args)
    set_seeds(random.randint(0, 1000))

    train(args, cfg)
