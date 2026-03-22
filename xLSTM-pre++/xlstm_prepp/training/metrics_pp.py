"""Metrics for TopologyLite GAT-xLSTM-K++."""

from __future__ import annotations

from typing import Optional

import torch

from xlstm_prepp.map_geometry import compute_trajectory_safety


SAFETY_KEYS = [
    "agent_size",
    "hard_polygon_vertices",
    "hard_polygon_vertex_mask",
    "hard_polygon_mask",
    "waypoint_segments",
    "waypoint_segment_mask",
    "hard_segments",
    "hard_segment_mask",
    "map_meta",
]


def _ade_k(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred - gt.unsqueeze(1), dim=-1).mean(dim=-1)


def _fde_k(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.norm(pred[:, :, -1, :] - gt[:, -1, :].unsqueeze(1), dim=-1)


def select_mode_indices(mode_logits: Optional[torch.Tensor]) -> torch.Tensor:
    if mode_logits is None:
        return torch.zeros(0, dtype=torch.long)
    return torch.argmax(mode_logits, dim=1)


def compute_selected_ade_fde(pred: torch.Tensor, gt: torch.Tensor, mode_indices: torch.Tensor):
    batch = pred.shape[0]
    arange_idx = torch.arange(batch, device=pred.device)
    selected = pred[arange_idx, mode_indices]
    ade = torch.norm(selected - gt, dim=-1).mean(dim=-1).mean()
    fde = torch.norm(selected[:, -1, :] - gt[:, -1, :], dim=-1).mean()
    return ade, fde


def compute_minade_k(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return _ade_k(pred, gt).min(dim=1).values.mean()


def compute_minfde_k(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return _fde_k(pred, gt).min(dim=1).values.mean()


def compute_mr_k(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    min_fde = _fde_k(pred, gt).min(dim=1).values
    return (min_fde > threshold).float().mean()


class MultiModalMetricsCalculator:
    def __init__(self, miss_threshold: float = 2.0, safety_margin: float = 1.5):
        self.miss_threshold = miss_threshold
        self.safety_margin = safety_margin
        self.reset()

    def reset(self):
        self.total_samples = 0
        self.sum_top1_ade = 0.0
        self.sum_top1_fde = 0.0
        self.sum_minade_k = 0.0
        self.sum_minfde_k = 0.0
        self.sum_mr_k = 0.0
        self.sum_intrusion_top1 = 0.0
        self.sum_min_intrusion = 0.0
        self.has_safety = False

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mode_logits: Optional[torch.Tensor] = None, safety_inputs: Optional[dict] = None):
        batch_size = pred.shape[0]
        if mode_logits is None:
            top1_idx = torch.zeros(batch_size, dtype=torch.long, device=pred.device)
        else:
            top1_idx = torch.argmax(mode_logits, dim=1)

        top1_ade, top1_fde = compute_selected_ade_fde(pred, gt, top1_idx)
        minade_k = compute_minade_k(pred, gt)
        minfde_k = compute_minfde_k(pred, gt)
        mr_k = compute_mr_k(pred, gt, threshold=self.miss_threshold)

        self.sum_top1_ade += top1_ade.item() * batch_size
        self.sum_top1_fde += top1_fde.item() * batch_size
        self.sum_minade_k += minade_k.item() * batch_size
        self.sum_minfde_k += minfde_k.item() * batch_size
        self.sum_mr_k += mr_k.item() * batch_size
        self.total_samples += batch_size

        if safety_inputs is None:
            return

        self.has_safety = True
        safety = compute_trajectory_safety(
            trajectory=pred,
            agent_size=safety_inputs["agent_size"],
            hard_polygon_vertices=safety_inputs["hard_polygon_vertices"],
            hard_polygon_vertex_mask=safety_inputs["hard_polygon_vertex_mask"],
            hard_polygon_mask=safety_inputs["hard_polygon_mask"],
            waypoint_segments=safety_inputs["waypoint_segments"],
            waypoint_segment_mask=safety_inputs["waypoint_segment_mask"],
            hard_segments=safety_inputs["hard_segments"],
            hard_segment_mask=safety_inputs["hard_segment_mask"],
            map_meta=safety_inputs["map_meta"],
            safety_margin=self.safety_margin,
        )
        batch_index = torch.arange(batch_size, device=pred.device)
        self.sum_intrusion_top1 += safety["intrusion_rate"][batch_index, top1_idx].mean().item() * batch_size
        self.sum_min_intrusion += safety["intrusion_penalty"].min(dim=1).values.mean().item() * batch_size

    def compute(self) -> dict:
        if self.total_samples == 0:
            result = {"ADE@1": 0.0, "FDE@1": 0.0, "minADE@K": 0.0, "minFDE@K": 0.0, "MR@2m": 0.0}
            if self.has_safety:
                result.update({"Intrusion@1": 0.0, "minIntrusion@K": 0.0})
            return result

        result = {
            "ADE@1": self.sum_top1_ade / self.total_samples,
            "FDE@1": self.sum_top1_fde / self.total_samples,
            "minADE@K": self.sum_minade_k / self.total_samples,
            "minFDE@K": self.sum_minfde_k / self.total_samples,
            "MR@2m": self.sum_mr_k / self.total_samples,
        }
        if self.has_safety:
            result.update(
                {
                    "Intrusion@1": self.sum_intrusion_top1 / self.total_samples,
                    "minIntrusion@K": self.sum_min_intrusion / self.total_samples,
                }
            )
        return result
