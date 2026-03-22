"""Multimodal trajectory predictors for GAT-xLSTM++ / xLSTM++ / LSTM++."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from xlstm_prepp.map_geometry import build_local_topology_features, select_local_map_subset

from .embedding import InputEmbedding
from .lstm_encoder import LSTMEncoder
from .mlstm import mLSTMEncoder
from .slstm import sLSTMEncoder
from .social_gat import HeteroSceneGAT
from .topology_lite_encoder import LocalMapTokenEncoder, LocalTopologyLiteEncoder
from .traj_decoder import TrajectoryLSTMDecoder
from .xlstm_encoder import xLSTMEncoder, xLSTMEncoderV2, xLSTMEncoderV3


class XTrajMultiModalPredictor(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}
        self.input_dim = config.get("input_dim", 4)
        self.neighbor_input_dim = config.get("neighbor_input_dim", 4)
        self.embedding_dim = config.get("embedding_dim", 32)
        self.encoder_hidden = config.get("encoder_hidden", 64)
        self.decoder_hidden = config.get("decoder_hidden", 128)
        self.encoder_type = config.get("encoder_type", "xlstm_v3")
        self.num_heads = config.get("num_heads", 2)
        self.dropout = config.get("dropout", 0.2)
        self.gat_dropout = config.get("gat_dropout", 0.1)
        self.num_modes = config.get("num_modes", 6)
        self.use_gat = bool(config.get("use_gat", False))
        self.use_motion_feedback = bool(config.get("use_motion_feedback", True))
        self.use_map_signal = bool(config.get("use_map_signal", False))
        self.topology_k_waypoints = int(config.get("topology_k_waypoints", 8))
        self.topology_k_hard_segments = int(config.get("topology_k_hard_segments", 8))
        self.topology_k_polygons = int(config.get("topology_k_polygons", 3))
        self.topology_output_dim = int(config.get("topology_output_dim", 24))
        self.local_scene_radius = float(config.get("local_scene_radius", 20.0))

        self.embedding = InputEmbedding(self.input_dim, self.embedding_dim)
        self.encoder = self._build_encoder()
        self.encoder_dropout = nn.Dropout(self.dropout)

        if self.use_gat:
            self.neighbor_embedding = InputEmbedding(self.neighbor_input_dim, self.embedding_dim)
            self.context_dim = self.encoder_hidden * 2
            self.ego_token_proj = nn.Sequential(
                nn.Linear(self.encoder_hidden + 4, self.context_dim),
                nn.LayerNorm(self.context_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.context_dim, self.context_dim),
            )
            self.obstacle_token_proj = nn.Sequential(
                nn.Linear(self.encoder_hidden + 4, self.context_dim),
                nn.LayerNorm(self.context_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.context_dim, self.context_dim),
            )
            self.scene_gat = HeteroSceneGAT(hidden_dim=self.context_dim, num_heads=self.num_heads, dropout=self.gat_dropout)
        else:
            self.neighbor_embedding = None
            self.context_dim = self.encoder_hidden
            self.ego_token_proj = None
            self.obstacle_token_proj = None
            self.scene_gat = None

        self.topology_encoder = None
        self.map_token_encoder = None
        if self.use_map_signal and self.use_gat:
            self.map_token_encoder = LocalMapTokenEncoder(
                waypoint_dim=8,
                hard_dim=8,
                polygon_dim=8,
                hidden_dim=config.get("topology_hidden_dim", 32),
                token_dim=self.context_dim,
                dropout=self.dropout,
            )
            self.context_fusion = None
        elif self.use_map_signal:
            self.topology_encoder = LocalTopologyLiteEncoder(
                waypoint_dim=8,
                hard_dim=8,
                polygon_dim=8,
                hidden_dim=config.get("topology_hidden_dim", 32),
                output_dim=self.topology_output_dim,
                dropout=self.dropout,
            )
            self.context_fusion = nn.Sequential(
                nn.Linear(self.context_dim + self.topology_output_dim, self.context_dim),
                nn.LayerNorm(self.context_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.context_dim, self.context_dim),
            )
        else:
            self.context_fusion = None

        self.mode_queries = nn.Parameter(torch.randn(self.num_modes, self.context_dim) * 0.02)
        self.mode_classifier = nn.Sequential(
            nn.Linear(self.context_dim, self.context_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout),
            nn.Linear(self.context_dim, self.num_modes),
        )
        self.decoder = TrajectoryLSTMDecoder(
            context_dim=self.context_dim,
            hidden_dim=self.decoder_hidden,
            dropout=self.dropout,
            use_motion_feedback=self.use_motion_feedback,
        )

    def _build_encoder(self):
        if self.encoder_type == "lstm":
            return LSTMEncoder(self.embedding_dim, self.encoder_hidden)
        if self.encoder_type == "mlstm":
            return mLSTMEncoder(self.embedding_dim, self.encoder_hidden)
        if self.encoder_type == "xlstm":
            return xLSTMEncoder(self.embedding_dim, self.encoder_hidden, dropout=self.dropout)
        if self.encoder_type == "xlstm_v2":
            return xLSTMEncoderV2(self.embedding_dim, self.encoder_hidden, dropout=self.dropout)
        if self.encoder_type == "xlstm_v3":
            return xLSTMEncoderV3(self.embedding_dim, self.encoder_hidden, dropout=self.dropout)
        return sLSTMEncoder(self.embedding_dim, self.encoder_hidden)

    @staticmethod
    def _repeat_by_mode(tensor: torch.Tensor, num_modes: int) -> torch.Tensor:
        return tensor.unsqueeze(1).repeat(1, num_modes, *([1] * (tensor.ndim - 1))).reshape(-1, *tensor.shape[1:]).contiguous()

    def _compute_topology_context(
        self,
        position: torch.Tensor,
        hard_polygon_vertices: torch.Tensor,
        hard_polygon_vertex_mask: torch.Tensor,
        hard_polygon_mask: torch.Tensor,
        waypoint_segments: torch.Tensor,
        waypoint_segment_mask: torch.Tensor,
        hard_segments: torch.Tensor,
        hard_segment_mask: torch.Tensor,
        map_meta: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_map_signal or self.topology_encoder is None:
            return torch.zeros(position.shape[0], 0, device=position.device, dtype=position.dtype)

        local_map = select_local_map_subset(
            position=position,
            hard_polygon_vertices=hard_polygon_vertices,
            hard_polygon_vertex_mask=hard_polygon_vertex_mask,
            hard_polygon_mask=hard_polygon_mask,
            waypoint_segments=waypoint_segments,
            waypoint_segment_mask=waypoint_segment_mask,
            hard_segments=hard_segments,
            hard_segment_mask=hard_segment_mask,
            map_meta=map_meta,
            topk_waypoints=self.topology_k_waypoints,
            topk_hard_segments=self.topology_k_hard_segments,
            topk_polygons=self.topology_k_polygons,
            local_radius=self.local_scene_radius,
        )
        topology_inputs = build_local_topology_features(position, local_map)
        return self.topology_encoder(
            waypoint_features=topology_inputs["waypoint_features"],
            waypoint_mask=topology_inputs["waypoint_mask"],
            hard_features=topology_inputs["hard_features"],
            hard_mask=topology_inputs["hard_mask"],
            polygon_features=topology_inputs["polygon_features"],
            polygon_mask=topology_inputs["polygon_mask"],
        )

    def _compute_map_tokens(
        self,
        position: torch.Tensor,
        hard_polygon_vertices: torch.Tensor,
        hard_polygon_vertex_mask: torch.Tensor,
        hard_polygon_mask: torch.Tensor,
        waypoint_segments: torch.Tensor,
        waypoint_segment_mask: torch.Tensor,
        hard_segments: torch.Tensor,
        hard_segment_mask: torch.Tensor,
        map_meta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.use_map_signal or self.map_token_encoder is None:
            return {
                "soft_tokens": torch.zeros(position.shape[0], 0, self.context_dim, device=position.device, dtype=position.dtype),
                "soft_mask": torch.zeros(position.shape[0], 0, device=position.device, dtype=torch.bool),
                "hard_tokens": torch.zeros(position.shape[0], 0, self.context_dim, device=position.device, dtype=position.dtype),
                "hard_mask": torch.zeros(position.shape[0], 0, device=position.device, dtype=torch.bool),
            }

        local_map = select_local_map_subset(
            position=position,
            hard_polygon_vertices=hard_polygon_vertices,
            hard_polygon_vertex_mask=hard_polygon_vertex_mask,
            hard_polygon_mask=hard_polygon_mask,
            waypoint_segments=waypoint_segments,
            waypoint_segment_mask=waypoint_segment_mask,
            hard_segments=hard_segments,
            hard_segment_mask=hard_segment_mask,
            map_meta=map_meta,
            topk_waypoints=self.topology_k_waypoints,
            topk_hard_segments=self.topology_k_hard_segments,
            topk_polygons=self.topology_k_polygons,
            local_radius=self.local_scene_radius,
        )
        topology_inputs = build_local_topology_features(position, local_map)
        return self.map_token_encoder(
            waypoint_features=topology_inputs["waypoint_features"],
            waypoint_mask=topology_inputs["waypoint_mask"],
            hard_features=topology_inputs["hard_features"],
            hard_mask=topology_inputs["hard_mask"],
            polygon_features=topology_inputs["polygon_features"],
            polygon_mask=topology_inputs["polygon_mask"],
        )

    def forward(
        self,
        obs_traj: torch.Tensor,
        pred_len: int,
        neighbor_trajs: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        hard_polygon_vertices: Optional[torch.Tensor] = None,
        hard_polygon_vertex_mask: Optional[torch.Tensor] = None,
        hard_polygon_mask: Optional[torch.Tensor] = None,
        waypoint_segments: Optional[torch.Tensor] = None,
        waypoint_segment_mask: Optional[torch.Tensor] = None,
        hard_segments: Optional[torch.Tensor] = None,
        hard_segment_mask: Optional[torch.Tensor] = None,
        map_meta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size = obs_traj.shape[0]
        obs_len = obs_traj.shape[1]
        target_emb = self.embedding(obs_traj)
        h_target = self.encoder_dropout(self.encoder(target_emb))

        if self.use_gat:
            if neighbor_trajs is None or neighbor_mask is None:
                raise ValueError("neighbor_trajs and neighbor_mask are required when use_gat=True")
            num_neighbors = neighbor_trajs.shape[1]
            neighbor_flat = neighbor_trajs.reshape(batch_size * num_neighbors, obs_len, self.neighbor_input_dim)
            neighbor_emb = self.neighbor_embedding(neighbor_flat)
            h_neighbors = self.encoder(neighbor_emb).reshape(batch_size, num_neighbors, self.encoder_hidden)
            h_neighbors = self.encoder_dropout(h_neighbors)
            ego_last = obs_traj[:, -1, :]
            if obs_len >= 2:
                ego_delta = obs_traj[:, -1, :2] - obs_traj[:, -2, :2]
            else:
                ego_delta = torch.zeros_like(obs_traj[:, -1, :2])
            ego_state_features = torch.cat([ego_last[:, 2:4], ego_delta], dim=-1)
            ego_token = self.ego_token_proj(torch.cat([h_target, ego_state_features], dim=-1))

            neighbor_current = neighbor_trajs[:, :, -1, :]
            neighbor_rel_pos = neighbor_current[:, :, :2] - ego_last[:, None, :2]
            neighbor_distance = torch.norm(neighbor_rel_pos, dim=-1)
            radius_scale = max(self.local_scene_radius, 1.0)
            obstacle_feature_tail = torch.cat([neighbor_rel_pos / radius_scale, neighbor_current[:, :, 2:4]], dim=-1)
            obstacle_tokens = self.obstacle_token_proj(torch.cat([h_neighbors, obstacle_feature_tail], dim=-1))
            obstacle_mask = neighbor_mask.bool() & (neighbor_distance <= self.local_scene_radius)
            obstacle_tokens = obstacle_tokens.masked_fill(~obstacle_mask.unsqueeze(-1), 0.0)

            hard_tokens = None
            hard_mask = None
            soft_tokens = None
            soft_mask = None
            if self.use_map_signal:
                required = [
                    hard_polygon_vertices,
                    hard_polygon_vertex_mask,
                    hard_polygon_mask,
                    waypoint_segments,
                    waypoint_segment_mask,
                    hard_segments,
                    hard_segment_mask,
                    map_meta,
                ]
                if any(item is None for item in required):
                    raise ValueError("Map tensors are required when use_map_signal=True")
                map_tokens = self._compute_map_tokens(
                    position=ego_last[:, :2],
                    hard_polygon_vertices=hard_polygon_vertices,
                    hard_polygon_vertex_mask=hard_polygon_vertex_mask,
                    hard_polygon_mask=hard_polygon_mask,
                    waypoint_segments=waypoint_segments,
                    waypoint_segment_mask=waypoint_segment_mask,
                    hard_segments=hard_segments,
                    hard_segment_mask=hard_segment_mask,
                    map_meta=map_meta,
                )
                hard_tokens = map_tokens["hard_tokens"]
                hard_mask = map_tokens["hard_mask"]
                soft_tokens = map_tokens["soft_tokens"]
                soft_mask = map_tokens["soft_mask"]

            base_context, _ = self.scene_gat(
                ego_token=ego_token,
                obstacle_tokens=obstacle_tokens,
                obstacle_mask=obstacle_mask,
                hard_map_tokens=hard_tokens,
                hard_map_mask=hard_mask,
                soft_map_tokens=soft_tokens,
                soft_map_mask=soft_mask,
            )
        else:
            base_context = h_target

        if self.use_map_signal and not self.use_gat:
            required = [
                hard_polygon_vertices,
                hard_polygon_vertex_mask,
                hard_polygon_mask,
                waypoint_segments,
                waypoint_segment_mask,
                hard_segments,
                hard_segment_mask,
                map_meta,
            ]
            if any(item is None for item in required):
                raise ValueError("Map tensors are required when use_map_signal=True")
            topology_context = self._compute_topology_context(
                position=obs_traj[:, -1, :2],
                hard_polygon_vertices=hard_polygon_vertices,
                hard_polygon_vertex_mask=hard_polygon_vertex_mask,
                hard_polygon_mask=hard_polygon_mask,
                waypoint_segments=waypoint_segments,
                waypoint_segment_mask=waypoint_segment_mask,
                hard_segments=hard_segments,
                hard_segment_mask=hard_segment_mask,
                map_meta=map_meta,
            )
            base_context = self.context_fusion(torch.cat([base_context, topology_context], dim=-1))

        mode_context = base_context.unsqueeze(1) + self.mode_queries.unsqueeze(0)
        mode_logits = self.mode_classifier(base_context)

        last_pos = obs_traj[:, -1, :2]
        if obs_len >= 2:
            last_delta = obs_traj[:, -1, :2] - obs_traj[:, -2, :2]
        else:
            last_delta = torch.zeros_like(last_pos)

        flat_context = mode_context.reshape(batch_size * self.num_modes, self.context_dim)
        flat_last_pos = self._repeat_by_mode(last_pos, self.num_modes)
        flat_last_delta = self._repeat_by_mode(last_delta, self.num_modes)
        pred_pos_flat = self.decoder(flat_context, pred_len, flat_last_pos, flat_last_delta)
        pred_pos = pred_pos_flat.reshape(batch_size, self.num_modes, pred_len, 2)
        return pred_pos, None, mode_logits
