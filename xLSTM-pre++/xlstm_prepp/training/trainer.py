"""Trainer for xLSTM-pre++ variants."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def set_postfix(self, *args, **kwargs):
            return None
    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable, **kwargs)

from xlstm_prepp.data import get_model_type
from xlstm_prepp.training.losses_pp import TopologyLiteLoss
from xlstm_prepp.training.metrics_pp import MultiModalMetricsCalculator, SAFETY_KEYS


class Trainer:
    def __init__(self, model, config: Dict, device: str = "cuda"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model_type = get_model_type(config)
        self.pred_len = config.get("data", {}).get("pred_len", 100)

        train_config = config.get("training", {})
        self.num_epochs = train_config.get("num_epochs", 100)
        self.lr = train_config.get("learning_rate", 5e-4)
        self.weight_decay = train_config.get("weight_decay", 1e-4)
        self.optimizer_name = train_config.get("optimizer", "AdamW")
        self.optimizer_fused = train_config.get("optimizer_fused", "auto")
        self.use_amp = bool(train_config.get("amp", False) and device.startswith("cuda"))
        self.grad_clip = float(train_config.get("grad_clip", 1.0))

        log_config = config.get("logging", {})
        self.log_dir = log_config.get("log_dir", "logs/GAT-xLSTM-K-I++")
        self.checkpoint_dir = log_config.get("checkpoint_dir", "checkpoints/GAT-xLSTM-K-I++")
        self.save_interval = log_config.get("save_interval", 10)
        self.history_path = os.path.join(self.log_dir, "training_history.json")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        loss_config = config.get("loss", {})
        map_config = config.get("map", {})
        eval_config = config.get("eval", {})
        physics_config = config.get("physics", {})
        self.criterion = TopologyLiteLoss(
            lambda_inertia=loss_config.get("lambda_inertia", 0.5),
            lambda_cls=loss_config.get("lambda_cls", 0.5),
            lambda_end=loss_config.get("lambda_end", 0.0),
            lambda_ctrl_smooth=loss_config.get("lambda_ctrl_smooth", 0.0),
            winner_metric=loss_config.get("winner_metric", "blend"),
            winner_ade_weight=loss_config.get("winner_ade_weight", 0.5),
            tau_start=loss_config.get("tau_start", 2.0),
            tau_end=loss_config.get("tau_end", 0.3),
            tau_anneal_epochs=loss_config.get("tau_anneal_epochs", 100),
            detach_assignment_for_cls=loss_config.get("detach_assignment_for_cls", True),
            enable_hard_soft_cls=loss_config.get("enable_hard_soft_cls", False),
            cls_soft_weight=loss_config.get("cls_soft_weight", 1.0),
            cls_hard_weight=loss_config.get("cls_hard_weight", 0.0),
        )
        self.metrics = MultiModalMetricsCalculator(
            miss_threshold=eval_config.get("miss_threshold", 2.0),
            safety_margin=map_config.get("safety_margin", 1.5),
        )

        self.optimizer = self._build_optimizer()
        scheduler_config = train_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "CosineAnnealing")
        if scheduler_type == "CosineAnnealing":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get("T_max", self.num_epochs),
                eta_min=scheduler_config.get("eta_min", 5e-5),
            )
        else:
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 30),
                gamma=scheduler_config.get("gamma", 0.5),
            )

        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.current_epoch = 0
        self.best_score = float("inf")
        self.history = []

    def _resolve_optimizer_fused(self) -> Optional[bool]:
        if self.optimizer_name.lower() not in {"adam", "adamw"}:
            return None
        if self.optimizer_fused == "always":
            return True
        if self.optimizer_fused == "never":
            return False
        return True if self.device.startswith("cuda") else None

    def _build_optimizer(self):
        fused = self._resolve_optimizer_fused()
        kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}
        if fused is not None:
            kwargs["fused"] = fused
        try:
            if self.optimizer_name.lower() == "adam":
                return Adam(self.model.parameters(), **kwargs)
            return AdamW(self.model.parameters(), **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            if self.optimizer_name.lower() == "adam":
                return Adam(self.model.parameters(), **kwargs)
            return AdamW(self.model.parameters(), **kwargs)

    def _build_safety_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: batch[key] for key in SAFETY_KEYS if key in batch}

    def _autocast(self):
        try:
            return torch.amp.autocast("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=self.use_amp)

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        non_blocking = self.device.startswith("cuda")
        return {key: value.to(self.device, non_blocking=non_blocking) for key, value in batch.items()}

    def _forward_batch(self, batch: Dict[str, torch.Tensor]):
        if self.model_type in {"topology_lite_xtrack_gat", "xtrack_gat_pp"}:
            return self.model(
                batch["obs_motion"],
                batch["init_state"],
                batch["neighbor_trajs"],
                batch["neighbor_mask"],
                self.pred_len,
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
        if self.model_type == "xtraj_gat_pp":
            return self.model(
                batch["obs_traj"],
                self.pred_len,
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
        if self.model_type == "xtraj_pp":
            return self.model(
                batch["obs_traj"],
                self.pred_len,
                hard_polygon_vertices=batch.get("hard_polygon_vertices"),
                hard_polygon_vertex_mask=batch.get("hard_polygon_vertex_mask"),
                hard_polygon_mask=batch.get("hard_polygon_mask"),
                waypoint_segments=batch.get("waypoint_segments"),
                waypoint_segment_mask=batch.get("waypoint_segment_mask"),
                hard_segments=batch.get("hard_segments"),
                hard_segment_mask=batch.get("hard_segment_mask"),
                map_meta=batch.get("map_meta"),
            )
        raise ValueError(f"Unsupported model type in trainer: {self.model_type}")

    def _compute_loss_dict(
        self,
        batch: Dict[str, torch.Tensor],
        pred_pos: torch.Tensor,
        pred_motion: torch.Tensor,
        mode_logits: torch.Tensor,
        aux_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        return self.criterion.forward_with_components(
            pred_pos=pred_pos,
            gt_pos=batch["gt_pos"],
            pred_motion=pred_motion,
            obs_motion=batch.get("obs_motion"),
            mode_logits=mode_logits,
            safety_inputs=self._build_safety_inputs(batch),
            aux_outputs=aux_outputs,
        )

    @staticmethod
    def _unpack_forward_outputs(model_outputs):
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 4:
            pred_pos, pred_motion, mode_logits, aux_outputs = model_outputs
            return pred_pos, pred_motion, mode_logits, aux_outputs
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 3:
            pred_pos, pred_motion, mode_logits = model_outputs
            return pred_pos, pred_motion, mode_logits, {}
        raise ValueError("Unexpected model output structure")

    @staticmethod
    def _accumulate_metrics(accumulator: Dict[str, float], loss_dict: Dict[str, torch.Tensor]):
        for key, value in loss_dict.items():
            accumulator[key] = accumulator.get(key, 0.0) + float(value.detach().item())

    @staticmethod
    def _prefix_loss_metrics(total: Dict[str, float], prefix: str) -> Dict[str, float]:
        metrics = {f"{prefix}_loss": total.get("total", 0.0)}
        for key, value in total.items():
            if key == "total":
                continue
            metrics[f"{prefix}_{key}"] = value
        return metrics

    @staticmethod
    def _normalize_metrics(accumulator: Dict[str, float], num_batches: int) -> Dict[str, float]:
        return {key: value / max(num_batches, 1) for key, value in accumulator.items()}

    def _score(self, metrics: Dict) -> float:
        return metrics.get("minADE@K", float("inf")) + 0.2 * metrics.get("Intrusion@1", 0.0)

    def _save_history(self) -> None:
        with open(self.history_path, "w") as file_obj:
            json.dump(self.history, file_obj, indent=2)

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None) -> str:
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "best_score": self.best_score,
                "metrics": metrics or {},
                "config": self.config,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_score = checkpoint.get("best_score", float("inf"))

    def train_epoch(self, train_loader) -> Dict[str, float]:
        self.model.train()
        self.criterion.set_epoch(self.current_epoch)
        total = {}
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in progress_bar:
            batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred_pos, pred_motion, mode_logits, aux_outputs = self._unpack_forward_outputs(self._forward_batch(batch))
                loss_dict = self._compute_loss_dict(batch, pred_pos, pred_motion, mode_logits, aux_outputs=aux_outputs)
                loss = loss_dict["total"]

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss detected at epoch {self.current_epoch}")

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            self._accumulate_metrics(total, loss_dict)
            num_batches += 1
            progress_bar.set_postfix({"loss": f"{total['total'] / max(num_batches, 1):.4f}"})

        total = self._normalize_metrics(total, num_batches)
        return self._prefix_loss_metrics(total, "train")

    @torch.no_grad()
    def validate(self, data_loader) -> Dict[str, float]:
        self.model.eval()
        self.criterion.set_epoch(self.current_epoch)
        self.metrics.reset()
        total = {}
        num_batches = 0

        for batch in data_loader:
            batch = self._move_batch_to_device(batch)
            with self._autocast():
                pred_pos, pred_motion, mode_logits, aux_outputs = self._unpack_forward_outputs(self._forward_batch(batch))
                loss_dict = self._compute_loss_dict(batch, pred_pos, pred_motion, mode_logits, aux_outputs=aux_outputs)
            self._accumulate_metrics(total, loss_dict)
            self.metrics.update(pred_pos, batch["gt_pos"], mode_logits, safety_inputs=self._build_safety_inputs(batch))
            num_batches += 1

        metrics = self.metrics.compute()
        total = self._normalize_metrics(total, num_batches)
        metrics.update(self._prefix_loss_metrics(total, "val"))
        return metrics

    def train(self, train_loader, val_loader=None) -> None:
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch + 1
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader) if val_loader is not None else {}
            record = {"epoch": self.current_epoch, "lr": current_lr, **train_metrics, **val_metrics}
            self.history.append(record)
            self._save_history()

            if val_metrics:
                score = self._score(val_metrics)
                if score < self.best_score:
                    self.best_score = score
                    self.save_checkpoint("best.pth", metrics=val_metrics)

            if self.current_epoch % self.save_interval == 0:
                self.save_checkpoint(f"epoch_{self.current_epoch}.pth", metrics=val_metrics)

            self.scheduler.step()

        self.save_checkpoint("final.pth")
