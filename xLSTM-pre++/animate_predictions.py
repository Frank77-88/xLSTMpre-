#!/usr/bin/env python
"""Generate animated trajectory prediction visualizations."""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch

from xlstm_prepp.data import build_map_adapter, create_dataset, get_display_name, get_model_type, load_config, resolve_path
from xlstm_prepp.runtime import create_model

from visualize_predictions import draw_map, forward_model, load_checkpoint, resolve_dataset_index, unpack_model_outputs


def build_animation_output_path(project_root: Path, display_name: str, output_arg: Optional[str], visual_index: int) -> Path:
    if output_arg is None:
        output_dir = project_root / "visualizations" / display_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"sample_{visual_index}_animation.gif"

    resolved_output = Path(resolve_path(output_arg, str(project_root)))
    if resolved_output.suffix.lower() != ".gif":
        resolved_output = resolved_output.with_suffix(".gif")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    return resolved_output


def render_animation(
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
    fps: int,
    tail_len: int,
):
    with torch.no_grad():
        pred_pos, _, mode_logits, aux_outputs = unpack_model_outputs(forward_model(model, batch, pred_len, model_type))

    pred_pos_single = pred_pos[0].cpu()
    mode_logits_single = mode_logits[0].cpu()
    gt_pos = batch["gt_pos"][0].cpu()
    obs_traj = batch["obs_traj"][0, :, :2].cpu() if "obs_traj" in batch else None

    if "init_state" in batch:
        start_point = batch["init_state"][0, :2].cpu()
    else:
        start_point = batch["obs_traj"][0, -1, :2].cpu()

    future_disp = torch.norm(gt_pos - start_point.unsqueeze(0), dim=1).max().item() if gt_pos.shape[0] > 0 else 0.0
    if future_disp < stationary_threshold:
        print(f"提示: sample {visual_index} 的未来轨迹几乎静止，GT 可能与 Start 重合")

    requested_top_k = 1 if top_k_arg <= 0 else top_k_arg
    top_k = min(requested_top_k, pred_pos_single.shape[0])
    top_indices = torch.topk(mode_logits_single, k=top_k).indices.tolist()

    colors = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
    behavior_queries = aux_outputs.get("behavior_queries")

    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame_idx: int):
        ax.clear()
        draw_map(ax, adapter)

        if obs_traj is not None:
            obs_start = max(0, obs_traj.shape[0] - tail_len)
            ax.plot(
                obs_traj[obs_start:, 0],
                obs_traj[obs_start:, 1],
                color="gray",
                linewidth=1.2,
                linestyle=":",
                label="Obs",
                zorder=3,
            )

        gt_steps = min(frame_idx + 1, gt_pos.shape[0])
        ax.plot(
            gt_pos[:gt_steps, 0],
            gt_pos[:gt_steps, 1],
            color="royalblue",
            linewidth=2.0,
            label="GT",
            zorder=4,
        )
        ax.scatter([start_point[0].item()], [start_point[1].item()], color="black", s=30, label="Start", zorder=5)

        for rank, mode_index in enumerate(top_indices):
            traj = pred_pos_single[mode_index]
            pred_steps = min(frame_idx + 1, traj.shape[0])
            label_suffix = f"mode {mode_index}"
            if behavior_queries is not None:
                label_suffix = f"behavior {mode_index}"
            ax.plot(
                traj[:pred_steps, 0],
                traj[:pred_steps, 1],
                color=colors[rank % len(colors)],
                linewidth=1.8,
                label=f"Pred-{rank + 1} ({label_suffix})",
                zorder=20 + (top_k - rank),
            )

        title_suffix = f"sample {visual_index}"
        if skip_stationary:
            title_suffix += f" -> ds[{dataset_index}]"
        time_sec = (frame_idx + 1) * 0.04
        ax.set_title(f"{display_name} | {title_suffix} | t={time_sec:.2f}s")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        return []

    anim = FuncAnimation(fig, update, frames=pred_len, interval=max(int(1000 / max(fps, 1)), 1), blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"已保存动画: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Animate xLSTM-pre++ predictions")
    parser.add_argument("--config", type=str, required=True, help="Config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--sample_idx", type=int, default=120, help="Sample index within split")
    parser.add_argument("--skip_stationary", action="store_true", help="Skip nearly stationary samples before indexing")
    parser.add_argument("--top_k", type=int, default=3, help="How many modes to animate; values <= 0 mean top1")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second for GIF")
    parser.add_argument("--tail_len", type=int, default=30, help="Observed history tail length to display")
    parser.add_argument("--output", type=str, default=None, help="Optional output gif path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    args.config = resolve_path(args.config, str(project_root))
    args.checkpoint = resolve_path(args.checkpoint, str(project_root))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"

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
    dataset_index = resolve_dataset_index(dataset, args.sample_idx, args.skip_stationary, stationary_threshold)
    if args.skip_stationary:
        print(f"skip_stationary 已启用，sample {args.sample_idx} 实际使用数据集样本索引: {dataset_index}")
    batch = collate_fn([dataset[dataset_index]])
    batch = {key: value.to(args.device) for key, value in batch.items()}

    output_path = build_animation_output_path(project_root, display_name, args.output, args.sample_idx)
    render_animation(
        model=model,
        model_type=model_type,
        batch=batch,
        pred_len=pred_len,
        adapter=adapter,
        display_name=display_name,
        visual_index=args.sample_idx,
        dataset_index=dataset_index,
        top_k_arg=args.top_k,
        skip_stationary=args.skip_stationary,
        stationary_threshold=stationary_threshold,
        output_path=output_path,
        fps=args.fps,
        tail_len=args.tail_len,
    )


if __name__ == "__main__":
    main()
