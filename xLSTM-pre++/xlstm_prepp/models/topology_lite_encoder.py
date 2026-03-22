"""Lightweight local topology encoders for pooled context and map tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalTopologyLiteEncoder(nn.Module):
    def __init__(
        self,
        waypoint_dim: int = 8,
        hard_dim: int = 8,
        polygon_dim: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.waypoint_branch = self._build_branch(waypoint_dim, hidden_dim, dropout)
        self.hard_branch = self._build_branch(hard_dim, hidden_dim, dropout)
        self.polygon_branch = self._build_branch(polygon_dim, hidden_dim, dropout)
        fusion_dim = hidden_dim * 2 * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    @staticmethod
    def _build_branch(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _masked_mean(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_max(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if encoded.shape[1] == 0:
            return torch.zeros(encoded.shape[0], encoded.shape[-1], device=encoded.device, dtype=encoded.dtype)
        masked = encoded.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        pooled = masked.max(dim=1).values
        pooled[~mask.any(dim=1)] = 0.0
        return pooled

    def _encode_branch(self, features: torch.Tensor, mask: torch.Tensor, branch: nn.Sequential) -> torch.Tensor:
        if features.shape[1] == 0:
            return torch.zeros(features.shape[0], self.hidden_dim * 2, device=features.device, dtype=features.dtype)
        encoded = branch(features)
        mean_feat = self._masked_mean(encoded, mask)
        max_feat = self._masked_max(encoded, mask)
        return torch.cat([mean_feat, max_feat], dim=-1)

    def forward(
        self,
        waypoint_features: torch.Tensor,
        waypoint_mask: torch.Tensor,
        hard_features: torch.Tensor,
        hard_mask: torch.Tensor,
        polygon_features: torch.Tensor,
        polygon_mask: torch.Tensor,
    ) -> torch.Tensor:
        waypoint_ctx = self._encode_branch(waypoint_features, waypoint_mask, self.waypoint_branch)
        hard_ctx = self._encode_branch(hard_features, hard_mask, self.hard_branch)
        polygon_ctx = self._encode_branch(polygon_features, polygon_mask, self.polygon_branch)
        fused = torch.cat([waypoint_ctx, hard_ctx, polygon_ctx], dim=-1)
        return self.fusion(fused)


class LocalMapTokenEncoder(nn.Module):
    def __init__(
        self,
        waypoint_dim: int = 8,
        hard_dim: int = 8,
        polygon_dim: int = 8,
        hidden_dim: int = 32,
        token_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.waypoint_branch = self._build_branch(waypoint_dim, hidden_dim, dropout)
        self.hard_branch = self._build_branch(hard_dim, hidden_dim, dropout)
        self.polygon_branch = self._build_branch(polygon_dim, hidden_dim, dropout)
        self.waypoint_proj = self._build_proj(hidden_dim, token_dim, dropout)
        self.hard_proj = self._build_proj(hidden_dim, token_dim, dropout)
        self.polygon_proj = self._build_proj(hidden_dim, token_dim, dropout)

    @staticmethod
    def _build_branch(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _build_proj(hidden_dim: int, token_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )

    def _encode_tokens(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        branch: nn.Sequential,
        projection: nn.Sequential,
    ) -> torch.Tensor:
        if features.shape[1] == 0:
            return torch.zeros(features.shape[0], 0, self.token_dim, device=features.device, dtype=features.dtype)
        encoded = projection(branch(features))
        return encoded.masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        waypoint_features: torch.Tensor,
        waypoint_mask: torch.Tensor,
        hard_features: torch.Tensor,
        hard_mask: torch.Tensor,
        polygon_features: torch.Tensor,
        polygon_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        soft_tokens = self._encode_tokens(waypoint_features, waypoint_mask, self.waypoint_branch, self.waypoint_proj)
        hard_segment_tokens = self._encode_tokens(hard_features, hard_mask, self.hard_branch, self.hard_proj)
        polygon_tokens = self._encode_tokens(polygon_features, polygon_mask, self.polygon_branch, self.polygon_proj)
        hard_tokens = torch.cat([hard_segment_tokens, polygon_tokens], dim=1)
        hard_token_mask = torch.cat([hard_mask, polygon_mask], dim=1)
        return {
            "soft_tokens": soft_tokens,
            "soft_mask": waypoint_mask,
            "hard_tokens": hard_tokens,
            "hard_mask": hard_token_mask,
            "hard_segment_tokens": hard_segment_tokens,
            "hard_segment_mask": hard_mask,
            "polygon_tokens": polygon_tokens,
            "polygon_mask": polygon_mask,
        }
