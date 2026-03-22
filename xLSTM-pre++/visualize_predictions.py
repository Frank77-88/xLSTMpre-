#!/usr/bin/env python
"""Visualize xLSTM-pre++ trajectory predictions on the map."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from xlstm_prepp.data import build_map_adapter, create_dataset, get_display_name, get_model_type, load_config, resolve_path
from xlstm_prepp.runtime import create_model


def load_checkpoint(checkpoint_path: str, device: str):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def forward_model(model, batch, pred_len: int, model_type: str):
    if model_type in {"topology_lite_xtrack_gat", "xtrack_gat_pp"}:
        return model(
            batch["obs_motion"],
            batch["init_state"],
            batch["neighbor_trajs"],
            batch["neighbor_mask"],
            pred_len,
            batch["hard_polygon_vertices"],
            batch["hard_polygon_vertex_mask"],
            batch["hard_polygon_mask"],
            batch["waypoint_segments"],
            batch["waypoint_segment_mask"],
            batch["hard_segments"],
            batch["hard_segment_mask"],
            batch["map_meta"],
            batch["agent_size"],
            batch.get("slot_polygon_vertices"),
            batch.get("slot_polygon_vertex_mask"),
            batch.get("slot_polygon_mask"),
        )
    if model_type == "xtraj_gat_pp":
        return model(
            batch["obs_traj"],
            pred_len,
            batch["neighbor_trajs"],
            batch["neighbor_mask"],
            batch.get("hard_polygon_vertices"),
            batch.get("hard_polygon_vertex_mask"),
            batch.get("hard_polygon_mask"),
            batch.get("waypoint_segments"),
            batch.get("waypoint_segment_mask"),
            batch.get("hard_segments"),
            batch.get("hard_segment_mask"),
            batch.get("map_meta"),
        )
    if model_type == "xtraj_pp":
        return model(
            batch["obs_traj"],
            pred_len,
            hard_polygon_vertices=batch.get("hard_polygon_vertices"),
            hard_polygon_vertex_mask=batch.get("hard_polygon_vertex_mask"),
            hard_polygon_mask=batch.get("hard_polygon_mask"),
            waypoint_segments=batch.get("waypoint_segments"),
            waypoint_segment_mask=batch.get("waypoint_segment_mask"),
            hard_segments=batch.get("hard_segments"),
            hard_segment_mask=batch.get("hard_segment_mask"),
            map_meta=batch.get("map_meta"),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def unpack_model_outputs(model_outputs):
    if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 4:
        pred_pos, pred_motion, mode_logits, aux_outputs = model_outputs
        return pred_pos, pred_motion, mode_logits, aux_outputs
    if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 3:
        pred_pos, pred_motion, mode_logits = model_outputs
        return pred_pos, pred_motion, mode_logits, {}
    raise ValueError("Unexpected model output structure during visualization")


def draw_map(ax, adapter):
    for polygon in adapter.parking_polygons:
        xs = [point[0] for point in polygon] + [polygon[0][0]]
        ys = [point[1] for point in polygon] + [polygon[0][1]]
        ax.fill(xs, ys, color="khaki", alpha=0.18, zorder=0)
        ax.plot(xs, ys, color="goldenrod", linewidth=0.8, alpha=0.7, zorder=1)
    for polygon in adapter.obstacle_polygons:
        xs = [point[0] for point in polygon] + [polygon[0][0]]
        ys = [point[1] for point in polygon] + [polygon[0][1]]
        ax.fill(xs, ys, color="dimgray", alpha=0.3, zorder=1)
    for start, end in adapter.slot_divider_segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], color="slategray", linewidth=0.5, alpha=0.5, zorder=2)
    for start, end in adapter.waypoint_segments:
        ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=0.6, alpha=0.3, linestyle="--", zorder=2)
    ax.plot([0, adapter.map_x, adapter.map_x, 0, 0], [0, 0, adapter.map_y, adapter.map_y, 0], color="black", linewidth=1.0, alpha=0.6, zorder=3)
    ax.set_xlim(0, adapter.map_x)
    ax.set_ylim(0, adapter.map_y)
    ax.set_aspect("equal")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")


def resolve_dataset_index(dataset, visual_index: int, skip_stationary: bool, stationary_threshold: float) -> int:
    if not skip_stationary:
        if visual_index < 0 or visual_index >= len(dataset):
            raise IndexError(f"sample_idx={visual_index} 超出数据集范围 [0, {len(dataset) - 1}]")
        return visual_index

    matched_count = -1
    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        gt_pos = sample["gt_pos"]
        start_point = sample["init_state"][:2] if "init_state" in sample else sample["obs_traj"][-1, :2]
        future_disp = torch.norm(gt_pos - start_point.unsqueeze(0), dim=1).max().item() if gt_pos.shape[0] > 0 else 0.0
        if future_disp >= stationary_threshold:
            matched_count += 1
            if matched_count == visual_index:
                return dataset_index
    raise ValueError(f"未找到第 {visual_index} 个非静止样本，请降低 sample_idx 或关闭 --skip_stationary")


def build_output_path(project_root: Path, display_name: str, output_arg: str, visual_index: int, batch_mode: bool) -> Path:
    if output_arg is None:
        output_dir = project_root / "visualizations" / display_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"sample_{visual_index}.png"

    resolved_output = Path(resolve_path(output_arg, str(project_root)))
    if batch_mode:
        resolved_output.mkdir(parents=True, exist_ok=True)
        return resolved_output / f"sample_{visual_index}.png"

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    return resolved_output


def render_sample(
    model,
    model_type: str,
    batch,
    pred_len: int,
    adapter,
    display_name: str,
    visual_index: int,
    dataset_index: int,
    top_k_arg: int,
    skip_stationary: bool,
    stationary_threshold: float,
    output_path: Path,
):
    with torch.no_grad():
        pred_pos, _, mode_logits, aux_outputs = unpack_model_outputs(forward_model(model, batch, pred_len, model_type))

    pred_pos_single = pred_pos[0]
    mode_logits_single = mode_logits[0]
    gt_pos = batch["gt_pos"][0].cpu()
    obs_traj = batch["obs_traj"][0, :, :2].cpu() if "obs_traj" in batch else None

    if "init_state" in batch:
        start_point = batch["init_state"][0, :2].cpu()
    else:
        start_point = batch["obs_traj"][0, -1, :2].cpu()

    requested_top_k = top_k_arg
    if requested_top_k <= 0:
        requested_top_k = 1
    top_k = min(requested_top_k, pred_pos_single.shape[0])
    top_indices = torch.topk(mode_logits_single, k=top_k).indices.tolist()
    pred_pos = pred_pos_single.cpu()
    mode_logits = mode_logits_single.cpu()

    future_disp = torch.norm(gt_pos - start_point.unsqueeze(0), dim=1).max().item() if gt_pos.shape[0] > 0 else 0.0
    if future_disp < stationary_threshold:
        print(f"提示: sample {visual_index} 的未来轨迹几乎静止，GT 可能与 Start 重合")

    fig, ax = plt.subplots(figsize=(10, 6))
    draw_map(ax, adapter)
    if obs_traj is not None:
        ax.plot(obs_traj[:, 0], obs_traj[:, 1], color="gray", linewidth=1.2, linestyle=":", label="Obs", zorder=3)
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], color="royalblue", linewidth=2.0, label="GT", zorder=4)
    ax.scatter([start_point[0].item()], [start_point[1].item()], color="black", s=30, label="Start", zorder=5)

    colors = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    behavior_queries = aux_outputs.get("behavior_queries")
    for rank, mode_index in enumerate(top_indices):
        traj = pred_pos[mode_index]
        line_zorder = 20 + (top_k - rank)
        label_suffix = f"mode {mode_index}"
        if behavior_queries is not None:
            label_suffix = f"behavior {mode_index}"
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=colors[rank % len(colors)],
            linewidth=1.5,
            label=f"Pred-{rank + 1} ({label_suffix})",
            zorder=line_zorder,
        )

    title_suffix = f"sample {visual_index}"
    if skip_stationary:
        title_suffix += f" -> ds[{dataset_index}]"
    ax.set_title(f"{display_name} | {title_suffix}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"已保存可视化: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize xLSTM-pre++ predictions")
    parser.add_argument("--config", type=str, required=True, help="Config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index within split")
    parser.add_argument("--sample_idx_end", type=int, default=None, help="Optional exclusive end index for batch export")
    parser.add_argument("--skip_stationary", action="store_true", help="Skip nearly stationary samples before indexing")
    parser.add_argument("--top_k", type=int, default=3, help="How many modes to plot; values <= 0 mean top1")
    parser.add_argument("--output", type=str, default=None, help="Optional output file for single export or directory for batch export")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    args.config = resolve_path(args.config, str(project_root))
    args.checkpoint = resolve_path(args.checkpoint, str(project_root))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"

    if args.sample_idx_end is not None and args.sample_idx_end <= args.sample_idx:
        raise ValueError("sample_idx_end 必须大于 sample_idx；区间采用左闭右开 [sample_idx, sample_idx_end)")

    config = load_config(args.config)
    display_name = get_display_name(config)
    model_type = get_model_type(config)
    dataset, collate_fn = create_dataset(config, split=args.split)
    adapter = build_map_adapter(config)

    model = create_model(config)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    pred_len = config.get("data", {}).get("pred_len", 100)
    stationary_threshold = max(float(config.get("data", {}).get("min_future_displacement", 0.0)), 1e-3)
    batch_mode = args.sample_idx_end is not None
    visual_indices = range(args.sample_idx, args.sample_idx_end) if batch_mode else [args.sample_idx]

    if args.top_k <= 0:
        print(f"top_k={args.top_k} 视为 top1，可视化 1 条预测轨迹")

    for visual_index in visual_indices:
        dataset_index = resolve_dataset_index(dataset, visual_index, args.skip_stationary, stationary_threshold)
        if args.skip_stationary:
            print(f"skip_stationary 已启用，sample {visual_index} 实际使用数据集样本索引: {dataset_index}")
        batch = collate_fn([dataset[dataset_index]])
        batch = {key: value.to(args.device) for key, value in batch.items()}
        output_path = build_output_path(project_root, display_name, args.output, visual_index, batch_mode)
        render_sample(
            model=model,
            model_type=model_type,
            batch=batch,
            pred_len=pred_len,
            adapter=adapter,
            display_name=display_name,
            visual_index=visual_index,
            dataset_index=dataset_index,
            top_k_arg=args.top_k,
            skip_stationary=args.skip_stationary,
            stationary_threshold=stationary_threshold,
            output_path=output_path,
        )

    if batch_mode:
        print(f"批量导出完成，共生成 {args.sample_idx_end - args.sample_idx} 张图片")


if __name__ == "__main__":
    main()
