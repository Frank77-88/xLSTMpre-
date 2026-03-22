"""Local heterogeneous graph attention for ego-centric scene fusion."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroSceneGAT(nn.Module):
    """Ego-centric heterogeneous scene fusion with typed relations.

    Only the ego node is updated, while obstacle / hard-map / soft-map nodes
    provide messages. This keeps the module lightweight for parking-lot scenes.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.ModuleDict(
            {
                "ego": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "obstacle": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "hard_map": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "soft_map": nn.Linear(hidden_dim, hidden_dim, bias=False),
            }
        )
        self.value_proj = nn.ModuleDict(
            {
                "ego": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "obstacle": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "hard_map": nn.Linear(hidden_dim, hidden_dim, bias=False),
                "soft_map": nn.Linear(hidden_dim, hidden_dim, bias=False),
            }
        )
        self.type_embeddings = nn.ParameterDict(
            {
                "ego": nn.Parameter(torch.randn(hidden_dim) * 0.02),
                "obstacle": nn.Parameter(torch.randn(hidden_dim) * 0.02),
                "hard_map": nn.Parameter(torch.randn(hidden_dim) * 0.02),
                "soft_map": nn.Parameter(torch.randn(hidden_dim) * 0.02),
            }
        )
        self.relation_key = nn.ParameterDict(
            {
                "self": nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02),
                "ego_to_obstacle": nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02),
                "ego_to_hard_map": nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02),
                "ego_to_soft_map": nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02),
            }
        )
        self.relation_bias = nn.ParameterDict(
            {
                "self": nn.Parameter(torch.zeros(num_heads)),
                "ego_to_obstacle": nn.Parameter(torch.zeros(num_heads)),
                "ego_to_hard_map": nn.Parameter(torch.zeros(num_heads)),
                "ego_to_soft_map": nn.Parameter(torch.zeros(num_heads)),
            }
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

        for projection in [self.query_proj, *self.key_proj.values(), *self.value_proj.values(), self.out_proj]:
            nn.init.xavier_uniform_(projection.weight)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor.view(tensor.shape[0], self.num_heads, self.head_dim)
        return tensor.view(tensor.shape[0], tensor.shape[1], self.num_heads, self.head_dim)

    def _encode_tokens(self, tokens: torch.Tensor, token_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        typed_tokens = tokens + self.type_embeddings[token_type].view(1, *((1,) * (tokens.ndim - 2)), -1)
        keys = self._reshape_heads(self.key_proj[token_type](typed_tokens))
        values = self._reshape_heads(self.value_proj[token_type](typed_tokens))
        return keys, values

    def _append_context(
        self,
        tokens: list[torch.Tensor],
        masks: list[torch.Tensor],
        rel_keys: list[torch.Tensor],
        rel_bias: list[torch.Tensor],
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        relation_name: str,
    ) -> None:
        if key_tensor.shape[1] == 0:
            return
        tokens.append(torch.cat([key_tensor, value_tensor], dim=-1))
        masks.append(mask_tensor)
        rel_keys.append(
            self.relation_key[relation_name].view(1, 1, self.num_heads, self.head_dim).expand(
                key_tensor.shape[0], key_tensor.shape[1], -1, -1
            )
        )
        rel_bias.append(
            self.relation_bias[relation_name].view(1, 1, self.num_heads).expand(
                key_tensor.shape[0], key_tensor.shape[1], -1
            )
        )

    def forward(
        self,
        ego_token: torch.Tensor,
        obstacle_tokens: Optional[torch.Tensor] = None,
        obstacle_mask: Optional[torch.Tensor] = None,
        hard_map_tokens: Optional[torch.Tensor] = None,
        hard_map_mask: Optional[torch.Tensor] = None,
        soft_map_tokens: Optional[torch.Tensor] = None,
        soft_map_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = ego_token.shape[0]
        ego_typed = ego_token + self.type_embeddings["ego"].view(1, -1)
        query = self._reshape_heads(self.query_proj(ego_typed))

        all_tokens = []
        all_masks = []
        all_rel_keys = []
        all_rel_bias = []

        self_keys, self_values = self._encode_tokens(ego_token.unsqueeze(1), "ego")
        self._append_context(
            all_tokens,
            all_masks,
            all_rel_keys,
            all_rel_bias,
            self_keys,
            self_values,
            torch.ones(batch_size, 1, device=ego_token.device, dtype=torch.bool),
            "self",
        )

        if obstacle_tokens is not None and obstacle_mask is not None:
            obstacle_keys, obstacle_values = self._encode_tokens(obstacle_tokens, "obstacle")
            self._append_context(
                all_tokens,
                all_masks,
                all_rel_keys,
                all_rel_bias,
                obstacle_keys,
                obstacle_values,
                obstacle_mask.bool(),
                "ego_to_obstacle",
            )

        if hard_map_tokens is not None and hard_map_mask is not None:
            hard_keys, hard_values = self._encode_tokens(hard_map_tokens, "hard_map")
            self._append_context(
                all_tokens,
                all_masks,
                all_rel_keys,
                all_rel_bias,
                hard_keys,
                hard_values,
                hard_map_mask.bool(),
                "ego_to_hard_map",
            )

        if soft_map_tokens is not None and soft_map_mask is not None:
            soft_keys, soft_values = self._encode_tokens(soft_map_tokens, "soft_map")
            self._append_context(
                all_tokens,
                all_masks,
                all_rel_keys,
                all_rel_bias,
                soft_keys,
                soft_values,
                soft_map_mask.bool(),
                "ego_to_soft_map",
            )

        stacked = torch.cat(all_tokens, dim=1)
        keys, values = stacked.split(self.head_dim, dim=-1)
        token_mask = torch.cat(all_masks, dim=1)
        relation_keys = torch.cat(all_rel_keys, dim=1)
        relation_bias = torch.cat(all_rel_bias, dim=1)

        scores = ((query.unsqueeze(1) * (keys + relation_keys)).sum(dim=-1) * self.scale) + relation_bias
        scores = scores.masked_fill(~token_mask.unsqueeze(-1), float("-inf"))
        attention = F.softmax(scores, dim=1)
        attention = torch.where(token_mask.unsqueeze(-1), attention, torch.zeros_like(attention))
        attention = self.dropout(attention)

        message = (attention.unsqueeze(-1) * values).sum(dim=1).reshape(batch_size, self.hidden_dim)
        fused = self.out_norm(ego_token + self.dropout(self.out_proj(message)))
        fused = self.ffn_norm(fused + self.dropout(self.ffn(fused)))
        return fused, {
            "scene_attention": attention.transpose(1, 2),
            "scene_node_count": token_mask.float().sum(dim=1),
        }
