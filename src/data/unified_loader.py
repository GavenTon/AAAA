# I created this data loader by refering following great reseaches & github repos.
# VP: stochastic video generation https://github.com/edenton/svg
# MP: On human motion prediction using recurrent neural network https://github.com/wei-mao-2019/LearnTrajDep
#     Trajectron++ https://github.com/StanfordASL/Trajectron-plus-plus
#     Motion Indeterminacy Diffusion https://github.com/gutianpei/mid
# TP: Social GAN https://github.com/agrimgupta92/sgan
from pathlib import Path
from typing import List, Optional
import dill
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Subset
from yacs.config import CfgNode


from data.TP.preprocessing import dict_collate


def load_environment_dataset(
    cfg: CfgNode, split: str = "train", aug_scene: bool = False
):
    if cfg.DATA.TASK != "TP":
        raise NotImplementedError("Only TP task is supported by the unified loader.")

    from .TP.trajectron_dataset import EnvironmentDataset, hypers

    if "longer" in cfg.DATA.DATASET_NAME and split != "train":
        i = int(cfg.DATA.DATASET_NAME[-1])
        cfg.defrost()
        cfg.DATA.OBSERVE_LENGTH -= i
        cfg.DATA.DATASET_NAME = cfg.DATA.DATASET_NAME[:-8]
        cfg.freeze()

    if cfg.DATA.DATASET_NAME == "sdd" and split != "train":
        i = cfg.DATA.PREDICT_LENGTH - 12
        cfg.defrost()
        cfg.DATA.OBSERVE_LENGTH -= i
        cfg.freeze()

    if cfg.DATA.DATASET_NAME == "sdd" and split == "val":
        env_path = (
            Path(cfg.DATA.PATH)
            / cfg.DATA.TASK
            / "processed_data"
            / f"{cfg.DATA.DATASET_NAME}_test.pkl"
        )
    else:
        env_path = (
            Path(cfg.DATA.PATH)
            / cfg.DATA.TASK
            / "processed_data"
            / f"{cfg.DATA.DATASET_NAME}_{split}.pkl"
        )

    with open(env_path, "rb") as f:
        env = dill.load(f, encoding="latin1")

    dataset = EnvironmentDataset(
        env,
        state=hypers[cfg.DATA.TP.STATE],
        pred_state=hypers[cfg.DATA.TP.PRED_STATE],
        node_freq_mult=hypers["scene_freq_mult_train"],
        scene_freq_mult=hypers["node_freq_mult_train"],
        hyperparams=hypers,
        min_history_timesteps=(
            1 if cfg.DATA.TP.ACCEPT_NAN and split == "train" else cfg.DATA.OBSERVE_LENGTH - 1
        ),
        min_future_timesteps=cfg.DATA.PREDICT_LENGTH,
        augment=aug_scene,
        normalize_direction=cfg.DATA.NORMALIZED,
    )

    for node_type_dataset in dataset:
        if node_type_dataset.node_type == "PEDESTRIAN":
            dataset = node_type_dataset
            break

    return dataset


def load_cluster_model(
    cfg: CfgNode, cluster_model_path: Optional[Path] = None
):
    model_path = (
        Path(cluster_model_path)
        if cluster_model_path is not None
        else _default_cluster_model_path(cfg)
    )
    if not model_path.exists():
        raise FileNotFoundError(
            f"cluster model not found at {model_path}. Provide --cluster_model_path."
        )
    with open(model_path, "rb") as f:
        clustering_model = pickle.load(f)
    return clustering_model


def predict_cluster_from_dict(
    data_dict: dict,
    cluster_model,
    normalize_direction: bool,
):
    gt = data_dict["gt_st"].cpu().numpy()[0]
    if np.isnan(gt).any():
        return None

    if not normalize_direction:
        obs = data_dict["obs_st"].cpu().numpy()[0]
        base_direction = obs[-1, 2:4]
        angle = -np.arctan2(base_direction[1], base_direction[0])
        rotate = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        gt = gt @ rotate.T

    cluster = cluster_model.predict(gt.reshape(1, -1))[0]
    return int(cluster)

def _default_cluster_model_path(cfg: CfgNode) -> Path:
    return (
        Path("src")
        / "clustering"
        / "models"
        / f"{cfg.DATA.DATASET_NAME}_train_{cfg.MGF.CLUSTER_METHOD}_{str(cfg.MGF.CLUSTER_N)}.pkl"
    )


def _select_cluster_indices(
    dataset,
    cluster_id: int,
    cluster_model,
    normalize_direction: bool,
) -> List[int]:
    indices: List[int] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is None:
            continue
        data_dict = dict_collate([sample])
        gt = data_dict["gt_st"].cpu().numpy()[0]
        if np.isnan(gt).any():
            continue
        if not normalize_direction:
            obs = data_dict["obs_st"].cpu().numpy()[0]
            base_direction = obs[-1, 2:4]
            angle = -np.arctan2(base_direction[1], base_direction[0])
            rotate = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                dtype=np.float32,
            )
            gt = gt @ rotate.T
        cluster = cluster_model.predict(gt.reshape(1, -1))[0]
        if cluster == cluster_id:
            indices.append(idx)
    return indices

def unified_loader(
        cfg: CfgNode,
        rand=True,
        split="train",
        batch_size=None,
        aug_scene=False,
        cluster_id: Optional[int] = None,
        cluster_model_path: Optional[Path] = None,
) -> DataLoader:

    if cfg.DATA.TASK != "TP":

        raise NotImplementedError("Only TP task is supported by the unified loader.")

    dataset = load_environment_dataset(cfg, split=split, aug_scene=aug_scene)

    if cluster_id is not None:
        clustering_model = load_cluster_model(cfg, cluster_model_path)
        selected_indices = _select_cluster_indices(
            dataset,
            cluster_id=cluster_id,
            cluster_model=clustering_model,
            normalize_direction=cfg.DATA.NORMALIZED,
        )
        if len(selected_indices) == 0:
            raise ValueError(
                f"No samples found for cluster {cluster_id} in split '{split}'."
            )
        dataset = Subset(dataset, selected_indices)


    from .TP.preprocessing import dict_collate as seq_collate

    if batch_size is None:
            batch_size = cfg.DATA.BATCH_SIZE
    drop_last = split == "train"
    if cluster_id is not None:
        drop_last = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=rand,
        num_workers=cfg.DATA.NUM_WORKERS,
        collate_fn=seq_collate,
        drop_last=drop_last,
        pin_memory=True,
    )

    return loader
