#!/usr/bin/env python
"""Evaluate xLSTM-pre++ model variants."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from xlstm_prepp.data import create_dataset, get_display_name, get_model_type, load_config, resolve_path
from xlstm_prepp.runtime import create_model
from xlstm_prepp.training.metrics_pp import MultiModalMetricsCalculator, SAFETY_KEYS


def load_checkpoint(checkpoint_path: str, device: str):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def build_safety_inputs(batch):
    return {key: batch[key] for key in SAFETY_KEYS if key in batch}


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
    raise ValueError("Unexpected model output structure during evaluation")


def _format_horizon_label(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-6:
        return f"{int(round(seconds))}s"
    return f"{seconds:.1f}s"


def infer_max_eval_seconds(dt: float, pred_len: int) -> int:
    if dt <= 0:
        return 0
    return max(int(pred_len * dt + 1e-6), 0)


def build_eval_horizons(dt: float, pred_len: int, max_seconds: Optional[int] = None) -> List[Tuple[float, int]]:
    horizons: List[Tuple[float, int]] = []
    if dt <= 0:
        return horizons
    if max_seconds is None:
        max_seconds = infer_max_eval_seconds(dt=dt, pred_len=pred_len)
    for seconds in range(1, max_seconds + 1):
        steps = int(round(seconds / dt))
        if 1 <= steps <= pred_len:
            horizons.append((float(seconds), steps))
    return horizons


def compute_min_metrics_at_horizon(pred: torch.Tensor, gt: torch.Tensor, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_h = pred[:, :, :steps, :]
    gt_h = gt[:, :steps, :]
    displacement = torch.norm(pred_h - gt_h.unsqueeze(1), dim=-1)
    minade = displacement.mean(dim=-1).min(dim=1).values.mean()
    minfde = torch.norm(pred_h[:, :, -1, :] - gt_h[:, -1, :].unsqueeze(1), dim=-1).min(dim=1).values.mean()
    return minade, minfde


class HorizonMetricAccumulator:
    def __init__(self, dt: float, pred_len: int, max_seconds: Optional[int] = None):
        self.horizons = build_eval_horizons(dt=dt, pred_len=pred_len, max_seconds=max_seconds)
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.metric_sums: Dict[str, float] = {}

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        if not self.horizons:
            return
        batch_size = pred.shape[0]
        for seconds, steps in self.horizons:
            minade, minfde = compute_min_metrics_at_horizon(pred, gt, steps)
            label = _format_horizon_label(seconds)
            self.metric_sums[f"minADE@{label}"] = self.metric_sums.get(f"minADE@{label}", 0.0) + minade.item() * batch_size
            self.metric_sums[f"minFDE@{label}"] = self.metric_sums.get(f"minFDE@{label}", 0.0) + minfde.item() * batch_size
        self.total_samples += batch_size

    def compute(self) -> Dict[str, float]:
        if self.total_samples == 0:
            return {}
        return {key: value / self.total_samples for key, value in self.metric_sums.items()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate xLSTM-pre++ variants")
    parser.add_argument("--config", type=str, required=True, help="Config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output json path")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    project_root = Path(__file__).resolve().parent
    args.config = resolve_path(args.config, str(project_root))
    args.checkpoint = resolve_path(args.checkpoint, str(project_root))

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"

    config = load_config(args.config)
    model_type = get_model_type(config)
    print(f"加载配置: {args.config}")
    print(f"评估模型: {get_display_name(config)}")
    dataset, collate_fn = create_dataset(config, split=args.split)
    print(f"样本数: {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=config.get("eval", {}).get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("training", {}).get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    model = create_model(config)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    data_config = config.get("data", {})
    pred_len = data_config.get("pred_len", 100)
    dt = float(data_config.get("dt", config.get("physics", {}).get("dt", 0.04)))
    metrics = MultiModalMetricsCalculator(
        miss_threshold=config.get("eval", {}).get("miss_threshold", 2.0),
        safety_margin=config.get("map", {}).get("safety_margin", 1.5),
    )
    horizon_metrics = HorizonMetricAccumulator(dt=dt, pred_len=pred_len, max_seconds=None)

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            pred_pos, _, mode_logits, _ = unpack_model_outputs(forward_model(model, batch, pred_len, model_type))
            metrics.update(pred_pos, batch["gt_pos"], mode_logits, safety_inputs=build_safety_inputs(batch))
            horizon_metrics.update(pred_pos, batch["gt_pos"])

    results = metrics.compute()
    results.update(horizon_metrics.compute())
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.save_json:
        output_path = resolve_path(args.save_json, str(project_root))
        with open(output_path, "w") as file_obj:
            json.dump(results, file_obj, indent=2, ensure_ascii=False)
        print(f"已保存评估结果: {output_path}")


if __name__ == "__main__":
    main()
