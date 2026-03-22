#!/usr/bin/env python
"""Sliding-window multimodal prediction visualization for xLSTM-pre++."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from xlstm_prepp.data import build_map_adapter, get_display_name, get_model_type, load_config, resolve_path
from xlstm_prepp.data.preprocessing import compute_motion_params
from xlstm_prepp.runtime import create_model

from visualize_predictions import draw_map, forward_model, load_checkpoint, unpack_model_outputs

PROJECT_ROOT = Path(__file__).resolve().parent
DLP_ROOT = PROJECT_ROOT.parent / "dlp-dataset"
if str(DLP_ROOT) not in sys.path:
    sys.path.insert(0, str(DLP_ROOT))

from dlp.dataset import Dataset as DlpDataset  # type: ignore  # noqa: E402


def draw_vehicle(
    ax,
    x: float,
    y: float,
    heading: float,
    color: str = "#00bfff",
    alpha: float = 1.0,
    width: float = 2.0,
    length: float = 4.5,
    zorder: int = 10,
):
    cx = x - (length / 2.0) * np.cos(heading)
    cy = y - (length / 2.0) * np.sin(heading)

    rect = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        facecolor=color,
        edgecolor="white",
        linewidth=1,
        alpha=alpha,
        zorder=zorder,
    )
    transform = Affine2D().rotate(heading).translate(cx, cy) + ax.transData
    rect.set_transform(transform)
    ax.add_patch(rect)
    return rect


def _sorted_instances(dlp: DlpDataset, agent_token: str) -> List[dict]:
    instances = dlp.get_agent_instances(agent_token)
    return sorted(instances, key=lambda inst: dlp.frames[inst["frame_token"]]["timestamp"])


def load_trajectory_bank(
    data_path: str,
    scene_list: Sequence[str],
    vehicle_types: Sequence[str],
    min_length: int,
) -> Tuple[List[Dict], Dict[str, Dict], Dict[str, List[Tuple[str, float, float, float, float]]]]:
    dlp = DlpDataset()
    for scene_name in scene_list:
        dlp.load(os.path.join(data_path, scene_name))

    long_trajs: List[Dict] = []
    all_trajs: Dict[str, Dict] = {}
    frame_to_vehicles: Dict[str, List[Tuple[str, float, float, float, float]]] = {}

    for agent_token, agent in dlp.agents.items():
        if str(agent["type"]) not in vehicle_types:
            continue

        instances = _sorted_instances(dlp, agent_token)
        if len(instances) < 10:
            continue

        traj_rows = []
        frame_tokens: List[str] = []
        for inst in instances:
            x, y = inst["coords"]
            heading = inst["heading"]
            speed = inst["speed"]
            frame_token = inst["frame_token"]
            traj_rows.append([x, y, heading, speed])
            frame_tokens.append(frame_token)
            frame_to_vehicles.setdefault(frame_token, []).append((agent_token, x, y, heading, speed))

        traj = np.asarray(traj_rows, dtype=np.float32)
        if len(traj) == 0:
            continue

        size = tuple(agent.get("size", (4.8, 2.0)))
        all_trajs[agent_token] = {
            "traj": traj,
            "frame_tokens": frame_tokens,
            "frame_token_to_index": {token: idx for idx, token in enumerate(frame_tokens)},
            "size": size,
        }

        if len(traj) < min_length:
            continue

        total_dist = float(np.sum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1)))
        if total_dist <= 10.0:
            continue

        long_trajs.append(
            {
                "agent_token": agent_token,
                "traj": traj,
                "frame_tokens": frame_tokens,
                "size": size,
                "total_dist": total_dist,
            }
        )

    long_trajs.sort(key=lambda item: item["total_dist"], reverse=True)
    return long_trajs, all_trajs, frame_to_vehicles


def get_neighbors_at_frame(
    frame_to_vehicles: Dict[str, List[Tuple[str, float, float, float, float]]],
    frame_token: str,
    target_agent: str,
    target_pos: np.ndarray,
    neighbor_distance: float,
) -> List[Dict]:
    neighbors: List[Dict] = []
    for agent_token, x, y, heading, speed in frame_to_vehicles.get(frame_token, []):
        if agent_token == target_agent:
            continue

        neighbor_pos = np.array([x, y], dtype=np.float32)
        distance = float(np.linalg.norm(neighbor_pos - target_pos))
        if distance <= neighbor_distance:
            neighbors.append(
                {
                    "agent_token": agent_token,
                    "x": x,
                    "y": y,
                    "heading": heading,
                    "speed": speed,
                    "distance": distance,
                }
            )
    neighbors.sort(key=lambda item: item["distance"])
    return neighbors


def get_agent_trajectory_at_frames(agent_data: Dict, ref_frame_tokens: Sequence[str]) -> Optional[np.ndarray]:
    local_index = agent_data["frame_token_to_index"]
    traj = agent_data["traj"]

    rows = []
    last_row = None
    for frame_token in ref_frame_tokens:
        row_idx = local_index.get(frame_token)
        if row_idx is None:
            if last_row is None:
                return None
            rows.append(last_row.copy())
            continue
        row = traj[row_idx].copy()
        rows.append(row)
        last_row = row

    if len(rows) != len(ref_frame_tokens):
        return None
    return np.asarray(rows, dtype=np.float32)


def build_neighbor_features(
    all_trajs: Dict[str, Dict],
    frame_to_vehicles: Dict[str, List[Tuple[str, float, float, float, float]]],
    target_agent: str,
    last_obs_frame: str,
    obs_frame_tokens: Sequence[str],
    target_pos: np.ndarray,
    num_neighbors: int,
    neighbor_distance: float,
    dt: float,
    traj_feature_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    neighbor_trajs = []
    neighbor_mask = []
    neighbors = get_neighbors_at_frame(
        frame_to_vehicles,
        last_obs_frame,
        target_agent,
        target_pos,
        neighbor_distance,
    )

    for slot in range(num_neighbors):
        if slot < len(neighbors):
            agent_token = neighbors[slot]["agent_token"]
            agent_data = all_trajs.get(agent_token)
            if agent_data is not None:
                n_traj = get_agent_trajectory_at_frames(agent_data, obs_frame_tokens)
                if n_traj is not None:
                    n_feat = np.zeros((len(obs_frame_tokens), 4), dtype=np.float32)
                    n_feat[:, 0] = n_traj[:, 0]
                    n_feat[:, 1] = n_traj[:, 1]
                    if traj_feature_mode == "vel_xy":
                        velocity = np.zeros((len(obs_frame_tokens), 2), dtype=np.float32)
                        if len(obs_frame_tokens) > 1:
                            velocity[1:] = (n_traj[1:, :2] - n_traj[:-1, :2]) / dt
                            velocity[0] = velocity[1]
                        n_feat[:, 2] = velocity[:, 0]
                        n_feat[:, 3] = velocity[:, 1]
                    else:
                        n_feat[:, 2] = n_traj[:, 3]
                        if len(obs_frame_tokens) > 1:
                            n_feat[1:, 3] = np.diff(n_traj[:, 3]) / dt
                            n_feat[0, 3] = n_feat[1, 3]
                    neighbor_trajs.append(n_feat)
                    neighbor_mask.append(1.0)
                    continue

        neighbor_trajs.append(np.zeros((len(obs_frame_tokens), 4), dtype=np.float32))
        neighbor_mask.append(0.0)

    return np.stack(neighbor_trajs, axis=0), np.asarray(neighbor_mask, dtype=np.float32)


def repeat_token_bank_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1).repeat(batch_size).clone()
    return tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.ndim)).contiguous().clone()


def prepare_model_input(
    traj_segment: np.ndarray,
    frame_tokens_segment: Sequence[str],
    target_agent: str,
    target_size: Tuple[float, float],
    all_trajs: Dict[str, Dict],
    frame_to_vehicles: Dict[str, List[Tuple[str, float, float, float, float]]],
    obs_len: int,
    num_neighbors: int,
    neighbor_distance: float,
    dt: float,
    device: torch.device,
    token_bank,
    traj_feature_mode: str,
) -> Dict[str, torch.Tensor]:
    params = compute_motion_params(traj_segment, dt)
    pos = params["pos"]
    v = params["v"]
    psi = params["psi"]
    ax = params["ax"]
    psi_dot = params["psi_dot"]

    if len(ax) < obs_len or len(psi_dot) < obs_len:
        raise ValueError(f"观测长度不足: ax={len(ax)}, psi_dot={len(psi_dot)}, obs_len={obs_len}")

    obs_motion = np.zeros((obs_len, 2), dtype=np.float32)
    obs_motion[:, 0] = ax[:obs_len]
    obs_motion[:, 1] = psi_dot[:obs_len]

    init_state = np.array(
        [
            pos[obs_len, 0],
            pos[obs_len, 1],
            v[obs_len],
            psi[obs_len],
        ],
        dtype=np.float32,
    )

    obs_frame_tokens = frame_tokens_segment[1 : obs_len + 1]
    last_obs_frame = frame_tokens_segment[obs_len]
    target_pos = pos[obs_len, :2].astype(np.float32)
    neighbor_trajs, neighbor_mask = build_neighbor_features(
        all_trajs=all_trajs,
        frame_to_vehicles=frame_to_vehicles,
        target_agent=target_agent,
        last_obs_frame=last_obs_frame,
        obs_frame_tokens=obs_frame_tokens,
        target_pos=target_pos,
        num_neighbors=num_neighbors,
        neighbor_distance=neighbor_distance,
        dt=dt,
        traj_feature_mode=traj_feature_mode,
    )

    batch = {
        "obs_motion": torch.from_numpy(obs_motion).unsqueeze(0).to(device),
        "init_state": torch.from_numpy(init_state).unsqueeze(0).to(device),
        "neighbor_trajs": torch.from_numpy(neighbor_trajs).unsqueeze(0).to(device),
        "neighbor_mask": torch.from_numpy(neighbor_mask).unsqueeze(0).to(device),
        "agent_size": torch.tensor([target_size], dtype=torch.float32, device=device),
    }

    for field in [
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
    ]:
        batch[field] = repeat_token_bank_tensor(getattr(token_bank, field), 1).to(device)
    return batch


def select_top_modes(pred_pos: np.ndarray, mode_logits: torch.Tensor, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    mode_probs = torch.softmax(mode_logits.squeeze(0), dim=0).cpu().numpy()
    order = np.argsort(-mode_probs)
    keep = order if top_k <= 0 else order[: min(top_k, len(order))]
    return pred_pos[keep], mode_probs[keep]


def constant_velocity_fallback(traj: np.ndarray, current_idx: int, pred_len: int, num_modes: int, dt: float) -> np.ndarray:
    current_pos = traj[current_idx, :2].astype(np.float32)
    current_heading = float(traj[current_idx, 2])
    current_speed = float(traj[current_idx, 3])

    steps = np.arange(1, pred_len + 1, dtype=np.float32)[:, None]
    direction = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)
    base_pred = current_pos[None, :] + steps * dt * current_speed * direction[None, :]
    return np.repeat(base_pred[None, :, :], num_modes, axis=0)


def format_prob_text(probs: np.ndarray, limit: int = 3) -> str:
    shown = probs[:limit]
    return ", ".join(f"{prob:.2f}" for prob in shown)


def resolve_output_path(project_root: Path, display_name: str, output_arg: Optional[str], save_gif: bool) -> Path:
    default_name = "sliding_prediction.gif" if save_gif else "sliding_prediction.png"
    if output_arg is None:
        output_dir = project_root / "visualizations" / display_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / default_name
    resolved = Path(resolve_path(output_arg, str(project_root)))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def compute_history_alpha(age: int, min_alpha: float = 0.08, base_alpha: float = 0.45, decay: float = 0.92) -> float:
    return max(min_alpha, base_alpha * (decay ** max(age, 0)))


def main():
    parser = argparse.ArgumentParser(description="xLSTM-pre++ sliding-window trajectory prediction visualization")
    parser.add_argument("--config", type=str, default="configs/GAT-xLSTM-K-F++.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/GAT-xLSTM-K-F++/best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--scene_limit", type=int, default=0, help="最多加载多少个场景，0 表示全部")
    parser.add_argument("--interval", type=int, default=80, help="动画帧间隔（毫秒）")
    parser.add_argument("--top_k", type=int, default=3, help="显示前 K 个模式，0 表示全部")
    parser.add_argument("--save_gif", action="store_true", help="保存为 GIF")
    parser.add_argument("--output", type=str, default=None, help="输出 GIF 路径")
    parser.add_argument("--traj_idx", type=int, default=0, help="选择第几条轨迹（按行驶距离排序）")
    parser.add_argument("--start_frame", type=int, default=0, help="从第几个滑窗帧开始可视化")
    parser.add_argument("--max_frames", type=int, default=120, help="最多导出多少个滑窗帧，0 表示全部")
    parser.add_argument(
        "--prediction_history",
        type=int,
        default=0,
        help="保留多少次历史预测，0 表示保留全部",
    )
    parser.add_argument("--list_trajs", action="store_true", help="列出可用轨迹")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    args.config = resolve_path(args.config, str(PROJECT_ROOT))
    args.checkpoint = resolve_path(args.checkpoint, str(PROJECT_ROOT))

    config = load_config(args.config)
    model_type = get_model_type(config)
    if model_type not in {"topology_lite_xtrack_gat", "xtrack_gat_pp"}:
        raise ValueError(f"该脚本只支持 xtrack 类模型，当前配置为: {model_type}")

    display_name = get_display_name(config)
    data_config = config.get("data", {})
    scene_key = {"train": "train_scenes", "val": "val_scenes", "test": "test_scenes"}[args.split]
    scene_list = list(data_config.get(scene_key, []))
    if args.scene_limit > 0:
        scene_list = scene_list[: args.scene_limit]
    if not scene_list:
        raise ValueError(f"配置中未找到 {scene_key}")

    obs_len = int(data_config.get("obs_len", 100))
    pred_len = int(data_config.get("pred_len", 100))
    dt = float(data_config.get("dt", config.get("physics", {}).get("dt", 0.04)))
    num_neighbors = int(data_config.get("num_neighbors", 4))
    neighbor_distance = float(data_config.get("neighbor_distance", data_config.get("local_scene_radius", 20.0)))
    vehicle_types = data_config.get("vehicle_types", ["Car", "Bus", "Truck", "Van"])
    traj_feature_mode = str(data_config.get("traj_feature_mode", "speed_ax")).lower()
    min_length = obs_len + pred_len + 2
    display_modes = config.get("model", {}).get("num_modes", 6) if args.top_k <= 0 else min(args.top_k, config.get("model", {}).get("num_modes", 6))

    adapter = build_map_adapter(config)
    token_bank = adapter.build_token_bank()

    print(f"加载配置: {args.config}")
    print(f"加载模型: {args.checkpoint}")
    print(f"使用场景: {scene_list}")

    model = create_model(config)
    checkpoint = load_checkpoint(args.checkpoint, str(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("搜索长轨迹...")
    long_trajs, all_trajs, frame_to_vehicles = load_trajectory_bank(
        data_path=resolve_path(data_config.get("dataset_path", "../dlp-dataset/data"), str(PROJECT_ROOT)),
        scene_list=scene_list,
        vehicle_types=vehicle_types,
        min_length=min_length,
    )
    if not long_trajs:
        raise ValueError("未找到足够长的轨迹")

    print(f"找到 {len(long_trajs)} 条长轨迹，共 {len(all_trajs)} 辆车")
    if args.list_trajs:
        print("\n可用轨迹列表:")
        for idx, item in enumerate(long_trajs[:20]):
            duration = len(item["traj"]) * dt
            print(f"  {idx}: {len(item['traj'])} 帧 ({duration:.1f}s), 距离: {item['total_dist']:.1f}m")
        return

    if args.traj_idx >= len(long_trajs):
        raise IndexError(f"轨迹索引 {args.traj_idx} 超出范围，最大 {len(long_trajs) - 1}")

    target = long_trajs[args.traj_idx]
    target_agent = target["agent_token"]
    traj = target["traj"]
    frame_tokens = target["frame_tokens"]
    print(f"选择轨迹: {len(traj)} 帧 ({len(traj) * dt:.1f}s), 行驶距离: {target['total_dist']:.1f}m")

    num_frames = len(traj) - min_length + 1
    if num_frames <= 0:
        raise ValueError("轨迹长度不足以进行滑动窗口预测")

    if args.start_frame < 0 or args.start_frame >= num_frames:
        raise IndexError(f"start_frame={args.start_frame} 超出范围 [0, {num_frames - 1}]")
    frame_end = num_frames if args.max_frames <= 0 else min(num_frames, args.start_frame + args.max_frames)
    frame_range = range(args.start_frame, frame_end)

    print(f"预计算滑动窗口预测... frame_range=[{args.start_frame}, {frame_end})")
    predictions = []
    success_count = 0
    fail_count = 0

    with torch.no_grad():
        for local_idx, start_idx in enumerate(frame_range):
            end_idx = start_idx + min_length
            traj_segment = traj[start_idx:end_idx]
            frame_tokens_segment = frame_tokens[start_idx:end_idx]

            try:
                batch = prepare_model_input(
                    traj_segment=traj_segment,
                    frame_tokens_segment=frame_tokens_segment,
                    target_agent=target_agent,
                    target_size=target["size"],
                    all_trajs=all_trajs,
                    frame_to_vehicles=frame_to_vehicles,
                    obs_len=obs_len,
                    num_neighbors=num_neighbors,
                    neighbor_distance=neighbor_distance,
                    dt=dt,
                    device=device,
                    token_bank=token_bank,
                    traj_feature_mode=traj_feature_mode,
                )
                pred_pos, _, mode_logits, _ = unpack_model_outputs(forward_model(model, batch, pred_len, model_type))
                pred_pos_np = pred_pos.squeeze(0).cpu().numpy()
                pred_keep, prob_keep = select_top_modes(pred_pos_np, mode_logits, args.top_k)
                predictions.append({"pred_pos": pred_keep, "mode_probs": prob_keep})
                success_count += 1

                if success_count == 1:
                    print(
                        f"  第一个预测成功: pred shape={pred_pos_np.shape}, "
                        f"top probs={format_prob_text(prob_keep, limit=min(3, len(prob_keep)))}"
                    )
            except Exception as exc:
                if fail_count == 0:
                    print(f"  预测失败，回退到匀速外推: {exc}")
                fallback = constant_velocity_fallback(
                    traj=traj,
                    current_idx=start_idx + obs_len,
                    pred_len=pred_len,
                    num_modes=display_modes,
                    dt=dt,
                )
                probs = np.zeros(display_modes, dtype=np.float32)
                probs[0] = 1.0
                predictions.append({"pred_pos": fallback, "mode_probs": probs})
                fail_count += 1

            if (local_idx + 1) % 50 == 0:
                print(f"  {local_idx + 1}/{len(frame_range)}")

    print(f"预测成功: {success_count}, 失败: {fail_count}")
    print(f"预计算完成，共 {len(predictions)} 帧")

    fig, ax = plt.subplots(figsize=(14, 8))
    draw_map(ax, adapter)

    mode_colors = ["#ff4444", "#ff8a3d", "#d264ff", "#7f66ff", "#ff66ad", "#8d99ae"]
    pred_lines = []
    for mode_idx in range(display_modes):
        line, = ax.plot(
            [],
            [],
            "--",
            color=mode_colors[mode_idx % len(mode_colors)],
            linewidth=2.7 if mode_idx == 0 else 1.8,
            alpha=0.95 if mode_idx == 0 else 0.55,
            label=f"Mode {mode_idx + 1}",
            zorder=5 - min(mode_idx, 2),
        )
        pred_lines.append(line)

    history_pred_artists = []
    gt_line, = ax.plot([], [], "-", color="#44ff44", linewidth=2.0, label="Ground Truth", alpha=0.65, zorder=4)
    history_line, = ax.plot([], [], "-", color="#00bfff", linewidth=1.5, alpha=0.3, label="History", zorder=3)
    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        color="black",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )

    vehicle_patches = []
    ax.legend(loc="upper right", facecolor="white", edgecolor="#cccccc", labelcolor="black")
    ax.set_xlabel("X (m)", color="black")
    ax.set_ylabel("Y (m)", color="black")
    ax.set_title(f"{display_name} Sliding Window Prediction", color="black", fontsize=14)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    fig.patch.set_facecolor("#ffffff")

    def init():
        for line in pred_lines:
            line.set_data([], [])
        gt_line.set_data([], [])
        history_line.set_data([], [])
        info_text.set_text("")
        return pred_lines + [gt_line, history_line, info_text]

    def update(frame: int):
        for patch in vehicle_patches:
            patch.remove()
        vehicle_patches.clear()

        for artist in history_pred_artists:
            artist.remove()
        history_pred_artists.clear()

        start_idx = args.start_frame + frame
        current_idx = obs_len + start_idx
        current_pos = traj[current_idx, :2]
        current_heading = traj[current_idx, 2]
        current_frame_token = frame_tokens[current_idx]

        target_rect = draw_vehicle(
            ax,
            current_pos[0],
            current_pos[1],
            current_heading,
            color="#00bfff",
            alpha=1.0,
            width=target["size"][1],
            length=target["size"][0],
            zorder=10,
        )
        vehicle_patches.append(target_rect)

        neighbors = get_neighbors_at_frame(
            frame_to_vehicles,
            current_frame_token,
            target_agent,
            current_pos,
            neighbor_distance,
        )
        for neighbor in neighbors[:num_neighbors]:
            neighbor_size = all_trajs.get(neighbor["agent_token"], {}).get("size", (4.8, 2.0))
            rect = draw_vehicle(
                ax,
                neighbor["x"],
                neighbor["y"],
                neighbor["heading"],
                color="#ffaa00",
                alpha=0.7,
                width=neighbor_size[1],
                length=neighbor_size[0],
                zorder=8,
            )
            vehicle_patches.append(rect)

        history_start = max(0, current_idx - 20)
        history = traj[history_start : current_idx + 1, :2]
        history_line.set_data(history[:, 0], history[:, 1])

        gt_future = traj[current_idx + 1 : current_idx + 1 + pred_len, :2]
        gt_display = np.vstack([current_pos, gt_future]) if len(gt_future) > 0 else current_pos[None, :]
        gt_line.set_data(gt_display[:, 0], gt_display[:, 1])

        ade = 0.0
        fde = 0.0
        prob_text = "-"

        pred_bundle = predictions[frame]
        pred_modes = pred_bundle["pred_pos"]
        probs = pred_bundle["mode_probs"]
        prob_text = format_prob_text(probs, limit=min(3, len(probs)))

        if args.prediction_history == 0:
            history_start_frame = 0
        else:
            history_start_frame = max(0, frame - args.prediction_history)

        for history_frame in range(history_start_frame, frame):
            history_start_idx = args.start_frame + history_frame
            history_current_idx = obs_len + history_start_idx
            history_current_pos = traj[history_current_idx, :2]
            history_bundle = predictions[history_frame]
            history_modes = history_bundle["pred_pos"]
            age = frame - history_frame
            alpha = compute_history_alpha(age)

            for mode_idx, mode_pred in enumerate(history_modes):
                artist, = ax.plot(
                    np.concatenate(([history_current_pos[0]], mode_pred[:, 0])),
                    np.concatenate(([history_current_pos[1]], mode_pred[:, 1])),
                    "--",
                    color=mode_colors[mode_idx % len(mode_colors)],
                    linewidth=1.4 if mode_idx == 0 else 1.0,
                    alpha=alpha if mode_idx == 0 else alpha * 0.8,
                    zorder=1,
                )
                history_pred_artists.append(artist)

        for mode_idx, line in enumerate(pred_lines):
            if mode_idx < len(pred_modes):
                pred_display = np.vstack([current_pos, pred_modes[mode_idx]])
                line.set_data(pred_display[:, 0], pred_display[:, 1])
            else:
                line.set_data([], [])

        gt_len = min(len(pred_modes[0]), len(gt_future)) if len(pred_modes) > 0 else 0
        if gt_len > 0:
            ade = float(np.mean(np.linalg.norm(pred_modes[0][:gt_len] - gt_future[:gt_len], axis=1)))
            fde = float(np.linalg.norm(pred_modes[0][gt_len - 1] - gt_future[gt_len - 1]))

        info_text.set_text(
            "\n".join(
                [
                    f"Time: {start_idx * dt:.2f}s",
                    f"Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f})",
                    f"Neighbors: {len(neighbors[:num_neighbors])} / {num_neighbors}",
                    f"Saved preds: {frame - history_start_frame + 1}",
                    f"Top probs: {prob_text}",
                    f"Top1 ADE: {ade:.2f}m | FDE: {fde:.2f}m",
                ]
            )
        )
        return history_pred_artists + pred_lines + [gt_line, history_line, info_text] + vehicle_patches

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(predictions),
        interval=args.interval,
        blit=False,
    )

    output_path = resolve_output_path(PROJECT_ROOT, display_name, args.output, args.save_gif)
    if args.save_gif:
        print(f"保存动画到: {output_path}")
        anim.save(output_path, writer=PillowWriter(fps=max(1, int(1000 / args.interval))))
        print("保存完成！")
    else:
        if len(predictions) > 0:
            update(len(predictions) - 1)
        fig.savefig(output_path, dpi=180)
        print(f"未启用 GIF，已保存当前画布到: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
