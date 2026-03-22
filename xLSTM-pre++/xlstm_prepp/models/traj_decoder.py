"""Autoregressive trajectory decoder for displacement-style predictors."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TrajectoryLSTMDecoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_motion_feedback: bool = True,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.use_motion_feedback = use_motion_feedback

        extra_dim = 4 if use_motion_feedback else 0
        self.init_h = nn.Linear(context_dim, hidden_dim)
        self.init_c = nn.Linear(context_dim, hidden_dim)
        self.cell = nn.LSTMCell(context_dim + extra_dim, hidden_dim)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _heading(delta: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        return delta / norm

    def forward(
        self,
        context: torch.Tensor,
        pred_len: int,
        last_pos: torch.Tensor,
        last_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        hidden = torch.tanh(self.init_h(context))
        cell = torch.tanh(self.init_c(context))
        current_pos = last_pos
        prev_delta = last_delta if last_delta is not None else torch.zeros(batch_size, 2, device=context.device, dtype=context.dtype)

        outputs = []
        for _ in range(pred_len):
            if self.use_motion_feedback:
                heading = self._heading(prev_delta)
                decoder_input = torch.cat([context, prev_delta, heading], dim=-1)
            else:
                decoder_input = context
            hidden, cell = self.cell(decoder_input, (hidden, cell))
            delta = self.output_head(hidden)
            current_pos = current_pos + delta
            outputs.append(current_pos)
            prev_delta = delta
        return torch.stack(outputs, dim=1)
