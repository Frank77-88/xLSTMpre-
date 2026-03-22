"""Losses for multimodal xLSTM-pre++ variants."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyLiteLoss(nn.Module):
    def __init__(
        self,
        lambda_inertia: float = 0.5,
        lambda_cls: float = 0.5,
        lambda_end: float = 0.0,
        lambda_ctrl_smooth: float = 0.0,
        winner_metric: str = "blend",
        winner_ade_weight: float = 0.5,
        tau_start: float = 2.0,
        tau_end: float = 0.3,
        tau_anneal_epochs: int = 100,
        detach_assignment_for_cls: bool = True,
        enable_hard_soft_cls: bool = False,
        cls_soft_weight: float = 1.0,
        cls_hard_weight: float = 0.0,
    ):
        super().__init__()
        self.lambda_inertia = float(lambda_inertia)
        self.lambda_cls = float(lambda_cls)
        self.lambda_end = float(lambda_end)
        self.lambda_ctrl_smooth = float(lambda_ctrl_smooth)
        self.winner_metric = winner_metric
        self.winner_ade_weight = float(winner_ade_weight)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.tau_anneal_epochs = int(tau_anneal_epochs)
        self.detach_assignment_for_cls = bool(detach_assignment_for_cls)
        self.enable_hard_soft_cls = bool(enable_hard_soft_cls)
        self.cls_soft_weight = float(cls_soft_weight)
        self.cls_hard_weight = float(cls_hard_weight)
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = max(int(epoch), 0)

    def _current_tau(self) -> float:
        if self.tau_anneal_epochs <= 0:
            return max(self.tau_end, 1e-3)
        ratio = min(self.current_epoch / float(self.tau_anneal_epochs), 1.0)
        tau = self.tau_start + ratio * (self.tau_end - self.tau_start)
        return max(float(tau), 1e-3)

    def _time_weights(self, pred_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        time_index = torch.arange(1, pred_len + 1, device=device, dtype=dtype)
        base_weights = time_index / pred_len
        start_boost = torch.exp(-time_index / 5.0)
        weights = base_weights + start_boost
        weights = weights / weights.mean()
        return weights.view(1, 1, pred_len, 1)

    def _weighted_mse_per_mode(self, pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> torch.Tensor:
        weights = self._time_weights(pred_pos.shape[2], pred_pos.device, pred_pos.dtype)
        diff2 = (pred_pos - gt_pos.unsqueeze(1)) ** 2
        return (diff2 * weights).mean(dim=(2, 3))

    def _ade_fde_per_mode(self, pred_pos: torch.Tensor, gt_pos: torch.Tensor):
        displacement = torch.norm(pred_pos - gt_pos.unsqueeze(1), dim=-1)
        ade = displacement.mean(dim=-1)
        fde = torch.norm(pred_pos[:, :, -1, :] - gt_pos[:, -1, :].unsqueeze(1), dim=-1)
        return ade, fde

    @staticmethod
    def _endpoint_error_per_mode(pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> torch.Tensor:
        return torch.norm(pred_pos[:, :, -1, :] - gt_pos[:, -1, :].unsqueeze(1), dim=-1)

    def _mode_score(self, pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> torch.Tensor:
        ade, fde = self._ade_fde_per_mode(pred_pos, gt_pos)
        if self.winner_metric == "ade":
            return ade
        if self.winner_metric == "fde":
            return fde
        return self.winner_ade_weight * ade + (1.0 - self.winner_ade_weight) * fde

    @staticmethod
    def _soft_assignment(score: torch.Tensor, tau: float) -> torch.Tensor:
        scaled = -score / max(tau, 1e-3)
        scaled = scaled - scaled.max(dim=1, keepdim=True).values
        return F.softmax(scaled, dim=1)

    @staticmethod
    def _masked_soft_assignment(score: torch.Tensor, valid_mask: torch.Tensor, tau: float) -> torch.Tensor:
        masked_score = score.masked_fill(~valid_mask, 1e6)
        assignment = TopologyLiteLoss._soft_assignment(masked_score, tau)
        assignment = assignment * valid_mask.to(dtype=score.dtype)
        return assignment / assignment.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _inertia_per_mode(pred_motion: torch.Tensor, obs_motion: torch.Tensor) -> torch.Tensor:
        obs_ax_mean = obs_motion[:, :, 0].mean(dim=1, keepdim=True)
        pred_ax_mean = pred_motion[:, :, :, 0].mean(dim=2)
        return F.relu(-obs_ax_mean * pred_ax_mean)

    @staticmethod
    def _control_smoothness(pred_motion: torch.Tensor) -> torch.Tensor:
        if pred_motion.shape[2] <= 1:
            return pred_motion.new_zeros(pred_motion.shape[0], pred_motion.shape[1])
        delta = pred_motion[:, :, 1:, :] - pred_motion[:, :, :-1, :]
        return delta.abs().mean(dim=(2, 3))

    def _compute_losses(
        self,
        pred_pos: torch.Tensor,
        gt_pos: torch.Tensor,
        pred_motion: Optional[torch.Tensor],
        obs_motion: Optional[torch.Tensor],
        mode_logits: torch.Tensor,
        safety_inputs: Optional[Dict[str, torch.Tensor]] = None,
        aux_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        score = self._mode_score(pred_pos, gt_pos)
        mode_valid_mask = None
        if aux_outputs:
            mode_valid_mask = aux_outputs.get("mode_valid_mask")
        if mode_valid_mask is None:
            mode_valid_mask = torch.ones_like(score, dtype=torch.bool)
        else:
            mode_valid_mask = mode_valid_mask.bool()

        global_assignment = self._masked_soft_assignment(score, mode_valid_mask, self._current_tau())
        global_winner_idx = torch.argmin(score.masked_fill(~mode_valid_mask, 1e6), dim=1)
        assignment = global_assignment
        winner_idx = global_winner_idx

        hard_winner = F.one_hot(winner_idx, num_classes=assignment.shape[1]).to(dtype=assignment.dtype)

        mse_per_mode = self._weighted_mse_per_mode(pred_pos, gt_pos)
        loss_reg = (assignment * mse_per_mode).sum(dim=1).mean()

        cls_target = assignment.detach() if self.detach_assignment_for_cls else assignment
        masked_mode_logits = mode_logits.masked_fill(~mode_valid_mask, -1e9)
        loss_cls_soft = F.kl_div(F.log_softmax(masked_mode_logits, dim=1), cls_target, reduction="batchmean")
        loss_cls_hard = F.cross_entropy(masked_mode_logits, winner_idx)
        if self.enable_hard_soft_cls:
            loss_cls = self.cls_soft_weight * loss_cls_soft + self.cls_hard_weight * loss_cls_hard
        else:
            loss_cls = loss_cls_soft

        if self.lambda_end <= 0:
            loss_end = torch.zeros((), device=pred_pos.device, dtype=pred_pos.dtype)
        else:
            endpoint_per_mode = self._endpoint_error_per_mode(pred_pos, gt_pos)
            loss_end = (assignment * endpoint_per_mode).sum(dim=1).mean()

        if pred_motion is None or obs_motion is None or self.lambda_inertia <= 0:
            loss_inertia = torch.zeros((), device=pred_pos.device, dtype=pred_pos.dtype)
        else:
            inertia_per_mode = self._inertia_per_mode(pred_motion, obs_motion)
            progress = min(self.current_epoch / float(max(self.tau_anneal_epochs, 1)), 1.0)
            inertia_weight = (1.0 - progress) * assignment + progress * hard_winner
            loss_inertia = (inertia_weight * inertia_per_mode).sum(dim=1).mean()

        if pred_motion is None or self.lambda_ctrl_smooth <= 0:
            loss_ctrl_smooth = torch.zeros((), device=pred_pos.device, dtype=pred_pos.dtype)
        else:
            ctrl_smooth_per_mode = self._control_smoothness(pred_motion)
            loss_ctrl_smooth = (assignment * ctrl_smooth_per_mode).sum(dim=1).mean()

        total_loss = (
            loss_reg
            + self.lambda_end * loss_end
            + self.lambda_cls * loss_cls
            + self.lambda_inertia * loss_inertia
            + self.lambda_ctrl_smooth * loss_ctrl_smooth
        )
        return {
            "total": total_loss,
            "reg": loss_reg,
            "cls": loss_cls,
            "end": loss_end,
            "cls_soft": loss_cls_soft,
            "cls_hard": loss_cls_hard,
            "inertia": loss_inertia,
            "ctrl_smooth": loss_ctrl_smooth,
        }

    def forward(
        self,
        pred_pos: torch.Tensor,
        gt_pos: torch.Tensor,
        pred_motion: Optional[torch.Tensor],
        obs_motion: Optional[torch.Tensor],
        mode_logits: torch.Tensor,
        safety_inputs: Optional[Dict[str, torch.Tensor]] = None,
        aux_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self._compute_losses(
            pred_pos=pred_pos,
            gt_pos=gt_pos,
            pred_motion=pred_motion,
            obs_motion=obs_motion,
            mode_logits=mode_logits,
            safety_inputs=safety_inputs,
            aux_outputs=aux_outputs,
        )["total"]

    def forward_with_components(
        self,
        pred_pos: torch.Tensor,
        gt_pos: torch.Tensor,
        pred_motion: Optional[torch.Tensor],
        obs_motion: Optional[torch.Tensor],
        mode_logits: torch.Tensor,
        safety_inputs: Optional[Dict[str, torch.Tensor]] = None,
        aux_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._compute_losses(
            pred_pos=pred_pos,
            gt_pos=gt_pos,
            pred_motion=pred_motion,
            obs_motion=obs_motion,
            mode_logits=mode_logits,
            safety_inputs=safety_inputs,
            aux_outputs=aux_outputs,
        )
