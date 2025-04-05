import argparse
import os
import random
import time

import numpy as np
import torch
from yacs.config import CfgNode

import wandb


def load_config(args: argparse.Namespace) -> CfgNode:
    from default_params import _C as cfg

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg_ = cfg.clone()
    if os.path.isfile(args.config_file):
        conf = args.config_file
        print(f"Configuration file loaded from {conf}.")
        cfg_.merge_from_file(conf)
        cfg_.OUTPUT_DIR = os.path.join(
            cfg_.OUTPUT_DIR,
            os.path.splitext(conf)[0],
            f"{time.strftime('%m%d_%H%M',time.localtime(time.time()))}-{args.model_name}",
        )

    else:
        raise FileNotFoundError

    if cfg_.LOAD_TUNED and args.mode != "tune":
        cfg_ = load_tuned(args, cfg_)
    # cfg_.freeze()

    return cfg_


def load_tuned(args: argparse.Namespace, cfg: CfgNode) -> CfgNode:
    import optuna

    study_path = os.path.join(cfg.OUTPUT_DIR, "optuna.db")
    if not os.path.exists(study_path):
        return cfg

    study_path = os.path.join("sqlite:///", study_path)
    print("load params from optuna database")
    study = optuna.load_study(storage=study_path, study_name="my_opt")
    trial_dict = study.best_trial.params

    for key in list(trial_dict.keys()):
        if type(trial_dict[key]) == str:
            exec(f"cfg.{key} = '{trial_dict[key]}'")
        else:
            exec(f"cfg.{key} = {trial_dict[key]}")

    return cfg


def optimizer_to_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


def set_seeds(seed, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_wandb(args):
    scene_name = args.scene.upper()
    project_name = f"MGF-{scene_name}"
    attach_name = args.model_name
    args.model_name = (
        scene_name
        + "-"
        + (
            f"clusterGMM{str(args.cluster_n)}_{str(args.var_init)}"
            if args.clusterGMM
            else "flowchain"
        )
        + ("*_" if args.learnVAR else "_")
        + f"mse{str(args.w_mse)}"
        + (f"{attach_name}" if attach_name is not None else "")
    )

    wandb.init(
        project=project_name,
        name=args.model_name,
        config=args,
    )
