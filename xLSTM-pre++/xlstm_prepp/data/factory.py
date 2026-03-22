"""Dataset factory and config helpers for xLSTM-pre++ variants."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import yaml
from torch.utils.data import Dataset

from xlstm_prepp.data.dataset import MapTokenCollator, TrajectoryDatasetPrePP
from xlstm_prepp.data.map_adapter import MapAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT.parent / "x-track-dlp"
SUPPORTED_MODEL_TYPES = {
    "topology_lite_xtrack_gat",
    "xtrack_gat_pp",
    "xtraj_gat_pp",
    "xtraj_pp",
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file_obj:
        return yaml.safe_load(file_obj)


def resolve_path(path: str, *base_dirs: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    for base_dir in base_dirs:
        candidate = os.path.abspath(os.path.join(base_dir, path))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(base_dirs[0], path)) if base_dirs else os.path.abspath(path)


def get_model_type(config: dict) -> str:
    return config.get("model", {}).get("type", "topology_lite_xtrack_gat")


def get_display_name(config: dict) -> str:
    log_dir = config.get("logging", {}).get("log_dir")
    if log_dir:
        return os.path.basename(log_dir.rstrip("/"))
    return get_model_type(config)


def build_map_adapter(config: dict) -> MapAdapter:
    map_config = config.get("map", {})
    map_path = resolve_path(
        map_config.get("map_path", "../dlp-dataset/dlp/parking_map.yml"),
        str(PROJECT_ROOT),
        str(LEGACY_ROOT),
    )
    return MapAdapter(
        map_path=map_path,
        slot_divider_as_obstacle=map_config.get("slot_divider_as_obstacle", True),
        max_slot_divider_segments=map_config.get("max_slot_divider_segments", 160),
    )


def _dataset_mode(model_type: str) -> Tuple[str, bool]:
    if model_type in {"topology_lite_xtrack_gat", "xtrack_gat_pp"}:
        return "xtrack", True
    if model_type == "xtraj_gat_pp":
        return "xtraj", True
    if model_type == "xtraj_pp":
        return "xtraj", False
    raise ValueError(f"Unsupported model type: {model_type}")


def create_dataset(config: dict, split: str) -> Tuple[Dataset, callable]:
    model_type = get_model_type(config)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type for dataset creation: {model_type}")

    data_config = config.get("data", {})
    scene_key = {"train": "train_scenes", "val": "val_scenes", "test": "test_scenes"}[split]
    scene_list = data_config.get(scene_key, [])
    mode, include_neighbors = _dataset_mode(model_type)

    dataset = TrajectoryDatasetPrePP(
        data_path=resolve_path(data_config.get("dataset_path", "../dlp-dataset/data"), str(PROJECT_ROOT), str(LEGACY_ROOT)),
        scene_list=scene_list,
        obs_len=data_config.get("obs_len", 100),
        pred_len=data_config.get("pred_len", 100),
        dt=data_config.get("dt", 0.04),
        vehicle_types=data_config.get("vehicle_types"),
        mode=mode,
        include_neighbors=include_neighbors,
        num_neighbors=data_config.get("num_neighbors", 4),
        neighbor_distance=data_config.get("neighbor_distance", data_config.get("local_scene_radius", 20.0)),
        filter_reverse=data_config.get("filter_reverse", False),
        window_stride=data_config.get("window_stride", 10),
        min_future_displacement=data_config.get("min_future_displacement", 0.0),
        traj_feature_mode=data_config.get("traj_feature_mode", "speed_ax"),
    )

    token_bank = build_map_adapter(config).build_token_bank()
    return dataset, MapTokenCollator(token_bank)
