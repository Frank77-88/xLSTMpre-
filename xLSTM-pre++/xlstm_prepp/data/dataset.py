"""Native dataset for xLSTM-pre++.

This dataset keeps the useful trajectory and neighbor indexing logic from the
legacy project while emitting only the fields required by the simplified static-
map pipeline.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from xlstm_prepp.data.preprocessing import compute_motion_params

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DLP_ROOT = PROJECT_ROOT.parent / "dlp-dataset"
if str(DLP_ROOT) not in sys.path:
    sys.path.insert(0, str(DLP_ROOT))

from dlp.dataset import Dataset as DlpDataset  # type: ignore  # noqa: E402


class TrajectoryDatasetPrePP(Dataset):
    def __init__(
        self,
        data_path: str,
        scene_list: List[str],
        obs_len: int = 100,
        pred_len: int = 100,
        dt: float = 0.04,
        vehicle_types: Optional[List[str]] = None,
        mode: str = "xtraj",
        include_neighbors: bool = False,
        num_neighbors: int = 4,
        neighbor_distance: float = 10.0,
        filter_reverse: bool = False,
        window_stride: int = 10,
        min_future_displacement: float = 0.0,
        traj_feature_mode: str = "speed_ax",
        cache_dir: Optional[str] = None,
    ):
        self.data_path = data_path
        self.scene_list = scene_list
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.dt = dt
        self.vehicle_types = vehicle_types or ["Car", "Bus", "Truck", "Van"]
        self.allowed_agent_types = [str(vehicle_type) for vehicle_type in self.vehicle_types]
        self.mode = mode.replace("_", "")
        self.include_neighbors = include_neighbors
        self.num_neighbors = num_neighbors
        self.neighbor_distance = neighbor_distance
        self.filter_reverse = filter_reverse
        self.window_stride = max(int(window_stride), 1)
        self.min_future_displacement = max(float(min_future_displacement), 0.0)
        self.traj_feature_mode = str(traj_feature_mode).lower()
        if self.traj_feature_mode not in {"speed_ax", "vel_xy"}:
            raise ValueError(f"Unsupported traj_feature_mode: {traj_feature_mode}")
        self.min_traj_len = obs_len + pred_len + 2

        if cache_dir is None:
            cache_dir = os.path.join(str(PROJECT_ROOT), "cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_key = self._compute_cache_key()
        cache_path = os.path.join(self.cache_dir, f"prepp_dataset_{cache_key}.pkl")
        if os.path.exists(cache_path):
            print(f"从缓存加载新数据集: {cache_path}")
            with open(cache_path, "rb") as file_obj:
                self.samples = pickle.load(file_obj)
            print(f"加载完成，共 {len(self.samples)} 个样本")
            return

        self.dlp_dataset = DlpDataset()
        self._load_scenes()
        self._build_indices()
        self.samples = self._build_samples()

        with open(cache_path, "wb") as file_obj:
            pickle.dump(self.samples, file_obj)

        del self.dlp_dataset
        del self.agent_trajectories
        if hasattr(self, "frame_to_vehicles"):
            del self.frame_to_vehicles

    def _compute_cache_key(self) -> str:
        key = (
            f"{sorted(self.scene_list)}_{self.obs_len}_{self.pred_len}_{self.dt}_"
            f"{self.mode}_{self.include_neighbors}_{self.num_neighbors}_{self.neighbor_distance}_{self.filter_reverse}_"
            f"{self.window_stride}_{self.min_future_displacement}_{self.traj_feature_mode}_static_v11"
        )
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _load_scenes(self) -> None:
        for scene_name in self.scene_list:
            scene_path = os.path.join(self.data_path, scene_name)
            self.dlp_dataset.load(scene_path)

    def _build_indices(self) -> None:
        self.agent_trajectories: Dict[str, List] = {}
        self.agent_sizes: Dict[str, tuple] = {}
        if self.include_neighbors:
            self.frame_to_vehicles: Dict[str, List] = {}

        for agent_token, agent in self.dlp_dataset.agents.items():
            raw_type = str(agent["type"])
            if raw_type not in self.allowed_agent_types:
                continue

            self.agent_sizes[agent_token] = tuple(agent.get("size", (4.8, 2.0)))
            instances = self.dlp_dataset.get_agent_instances(agent_token)
            if len(instances) == 0:
                continue

            traj_data = []
            for instance in instances:
                x, y = instance["coords"]
                frame_token = instance["frame_token"]
                heading = instance["heading"]
                speed = instance["speed"]
                traj_data.append((frame_token, x, y, heading, speed))
                if self.include_neighbors:
                    self.frame_to_vehicles.setdefault(frame_token, []).append((agent_token, x, y, heading, speed))

            if traj_data:
                self.agent_trajectories[agent_token] = traj_data

    def _build_samples(self) -> List[Dict[str, torch.Tensor]]:
        samples: List[Dict[str, torch.Tensor]] = []
        total_agents = len(self.agent_trajectories)
        for agent_index, (agent_token, traj_data) in enumerate(self.agent_trajectories.items()):
            if agent_index % 100 == 0:
                print(f"  处理 agent {agent_index}/{total_agents}")

            if len(traj_data) < self.min_traj_len:
                continue

            frame_tokens = [item[0] for item in traj_data]
            trajectory = np.array([[item[1], item[2], item[3], item[4]] for item in traj_data], dtype=np.float32)

            num_samples = len(trajectory) - self.min_traj_len + 1
            for start_index in range(0, num_samples, self.window_stride):
                end_index = start_index + self.min_traj_len
                if end_index <= len(trajectory):
                    sample = self._create_sample(
                        traj_segment=trajectory[start_index:end_index],
                        frame_tokens=frame_tokens[start_index:end_index],
                        agent_token=agent_token,
                    )
                    if sample is not None:
                        samples.append(sample)
        return samples

    def _get_neighbors_at_frame(self, frame_token: str, target_pos: np.ndarray, target_agent_token: str) -> List[Dict]:
        vehicles = self.frame_to_vehicles.get(frame_token, [])
        neighbors = []
        for agent_token, x, y, heading, speed in vehicles:
            if agent_token == target_agent_token:
                continue
            neighbor_pos = np.array([x, y], dtype=np.float32)
            distance = np.linalg.norm(neighbor_pos - target_pos)
            if distance <= self.neighbor_distance:
                neighbors.append(
                    {
                        "agent_token": agent_token,
                        "distance": distance,
                        "heading": heading,
                        "speed": speed,
                    }
                )
        neighbors.sort(key=lambda item: item["distance"])
        return neighbors[: self.num_neighbors]

    def _get_neighbor_trajectory(self, agent_token: str, ref_frame_tokens: List[str]) -> Optional[np.ndarray]:
        traj_data = self.agent_trajectories.get(agent_token)
        if traj_data is None:
            return None
        frame_to_data = {item[0]: item for item in traj_data}

        traj = []
        for frame_token in ref_frame_tokens:
            if frame_token in frame_to_data:
                _, x, y, heading, speed = frame_to_data[frame_token]
                traj.append([x, y, heading, speed])
            elif traj:
                traj.append(traj[-1])
            else:
                return None

        if len(traj) != len(ref_frame_tokens):
            return None
        return np.array(traj, dtype=np.float32)

    def _has_reverse_motion(self, traj_segment: np.ndarray) -> bool:
        pos = traj_segment[:, :2]
        heading = traj_segment[:, 2]
        delta = pos[1:] - pos[:-1]
        vel_angle = np.arctan2(delta[:, 1], delta[:, 0])
        angle_diff = vel_angle - heading[:-1]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        reverse_threshold = np.pi * 2.0 / 3.0
        displacement = np.linalg.norm(delta, axis=1)
        moving_mask = displacement > 0.01
        if moving_mask.sum() == 0:
            return False
        reverse_ratio = (np.abs(angle_diff[moving_mask]) > reverse_threshold).mean()
        return reverse_ratio > 0.1

    @staticmethod
    def _future_motion_span(start_pos: np.ndarray, future_pos: np.ndarray) -> float:
        if len(future_pos) <= 1:
            return float(np.linalg.norm(future_pos[0] - start_pos)) if len(future_pos) == 1 else 0.0
        offsets = future_pos - start_pos[None, :]
        return float(np.linalg.norm(offsets, axis=1).max())

    def _build_obs_traj_features(self, pos: np.ndarray, speed: np.ndarray, ax: np.ndarray, obs_end: int) -> np.ndarray:
        obs_pos = pos[1 : obs_end + 1].astype(np.float32)
        obs_traj = np.zeros((self.obs_len, 4), dtype=np.float32)
        obs_traj[:, 0] = obs_pos[:, 0]
        obs_traj[:, 1] = obs_pos[:, 1]
        if self.traj_feature_mode == "vel_xy":
            velocity = np.zeros((self.obs_len, 2), dtype=np.float32)
            if self.obs_len > 1:
                velocity[1:] = (obs_pos[1:] - obs_pos[:-1]) / self.dt
                velocity[0] = velocity[1]
            obs_traj[:, 2] = velocity[:, 0]
            obs_traj[:, 3] = velocity[:, 1]
            return obs_traj

        obs_traj[:, 2] = speed[:obs_end]
        obs_traj[:-1, 3] = ax[: obs_end - 1]
        obs_traj[-1, 3] = ax[obs_end - 2] if obs_end > 1 else 0.0
        return obs_traj

    def _build_neighbor_features(self, neighbor_traj: np.ndarray) -> np.ndarray:
        features = np.zeros((self.obs_len, 4), dtype=np.float32)
        positions = neighbor_traj[:, :2].astype(np.float32)
        features[:, 0] = positions[:, 0]
        features[:, 1] = positions[:, 1]
        if self.traj_feature_mode == "vel_xy":
            velocity = np.zeros((self.obs_len, 2), dtype=np.float32)
            if self.obs_len > 1:
                velocity[1:] = (positions[1:] - positions[:-1]) / self.dt
                velocity[0] = velocity[1]
            features[:, 2] = velocity[:, 0]
            features[:, 3] = velocity[:, 1]
            return features

        features[:, 2] = neighbor_traj[:, 3]
        if self.obs_len > 1:
            features[1:, 3] = np.diff(neighbor_traj[:, 3]) / self.dt
            features[0, 3] = features[1, 3]
        return features

    def _create_sample(
        self,
        traj_segment: np.ndarray,
        frame_tokens: List[str],
        agent_token: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            if self.filter_reverse and self._has_reverse_motion(traj_segment):
                return None

            params = compute_motion_params(traj_segment, self.dt)
            pos = params["pos"]
            speed = params["v"]
            heading = params["psi"]
            ax = params["ax"]
            psi_dot = params["psi_dot"]
            obs_end = self.obs_len

            current_pos = pos[obs_end].astype(np.float32)
            future_pos = pos[obs_end + 1 : obs_end + 1 + self.pred_len].astype(np.float32)
            if self.min_future_displacement > 0.0:
                future_span = self._future_motion_span(current_pos, future_pos)
                if future_span < self.min_future_displacement:
                    return None

            sample: Dict[str, torch.Tensor] = {}
            sample["obs_traj"] = torch.from_numpy(self._build_obs_traj_features(pos, speed, ax, obs_end))

            if self.mode in ["xtrack", "both"]:
                ax_max = 9.0
                psi_dot_max = 1.5
                obs_ax = ax[:obs_end]
                obs_psi_dot = psi_dot[:obs_end]
                gt_ax = ax[obs_end : obs_end + self.pred_len]
                gt_psi_dot = psi_dot[obs_end : obs_end + self.pred_len]
                if (np.abs(obs_ax) > ax_max).any() or (np.abs(gt_ax) > ax_max).any():
                    return None
                if (np.abs(obs_psi_dot) > psi_dot_max).any() or (np.abs(gt_psi_dot) > psi_dot_max).any():
                    return None

                obs_motion = np.zeros((self.obs_len, 2), dtype=np.float32)
                obs_motion[:, 0] = obs_ax
                obs_motion[:, 1] = obs_psi_dot
                init_state = np.array(
                    [
                        pos[obs_end, 0],
                        pos[obs_end, 1],
                        speed[obs_end],
                        heading[obs_end],
                    ],
                    dtype=np.float32,
                )
                gt_motion = np.zeros((self.pred_len, 2), dtype=np.float32)
                gt_motion[:, 0] = gt_ax
                gt_motion[:, 1] = gt_psi_dot
                sample["obs_motion"] = torch.from_numpy(obs_motion)
                sample["init_state"] = torch.from_numpy(init_state)
                sample["gt_motion"] = torch.from_numpy(gt_motion)

            sample["gt_pos"] = torch.from_numpy(future_pos)
            sample["agent_size"] = torch.tensor(self.agent_sizes.get(agent_token, (4.8, 2.0)), dtype=torch.float32)

            last_obs_frame = frame_tokens[obs_end]
            target_pos = pos[obs_end, :2]

            if self.include_neighbors:
                neighbors = self._get_neighbors_at_frame(last_obs_frame, target_pos, agent_token)
                obs_frame_tokens = frame_tokens[1 : obs_end + 1]
                neighbor_trajs = []
                neighbor_mask = []
                for neighbor_index in range(self.num_neighbors):
                    if neighbor_index < len(neighbors):
                        neighbor = neighbors[neighbor_index]
                        neighbor_traj = self._get_neighbor_trajectory(neighbor["agent_token"], obs_frame_tokens)
                        if neighbor_traj is not None:
                            neighbor_trajs.append(self._build_neighbor_features(neighbor_traj))
                            neighbor_mask.append(1.0)
                            continue
                    neighbor_trajs.append(np.zeros((self.obs_len, 4), dtype=np.float32))
                    neighbor_mask.append(0.0)
                sample["neighbor_trajs"] = torch.from_numpy(np.stack(neighbor_trajs, axis=0))
                sample["neighbor_mask"] = torch.tensor(neighbor_mask, dtype=torch.float32)

            return sample
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]


class StackCollator:
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in batch[0].keys()}


class MapTokenCollator(StackCollator):
    def __init__(self, token_bank):
        self.token_bank = token_bank
        self.map_fields = [
            "map_meta",
            "slot_polygon_vertices",
            "slot_polygon_vertex_mask",
            "slot_polygon_mask",
            "hard_polygon_vertices",
            "hard_polygon_vertex_mask",
            "hard_polygon_mask",
            "waypoint_segments",
            "waypoint_segment_mask",
            "hard_segments",
            "hard_segment_mask",
        ]

    def _repeat(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor.reshape(1).repeat(batch_size).clone()
        expanded = tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.ndim))
        return expanded.contiguous().clone()

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        stacked = super().__call__(batch)
        batch_size = len(batch)
        for field in self.map_fields:
            stacked[field] = self._repeat(getattr(self.token_bank, field), batch_size)
        return stacked
