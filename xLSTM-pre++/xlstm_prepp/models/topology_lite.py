"""Map-aware GAT-xLSTM-K++ with local heterogeneous scene fusion."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm_prepp.map_geometry import build_local_topology_features, min_signed_distance_to_polygons, select_local_map_subset

from .embedding import InputEmbedding
from .lstm_encoder import LSTMEncoder
from .mlstm import mLSTMEncoder
from .slstm import sLSTMEncoder
from .social_gat import HeteroSceneGAT
from .topology_lite_decoder import TopologyLiteDecoder
from .topology_lite_encoder import LocalMapTokenEncoder
from .xlstm_encoder import xLSTMEncoder, xLSTMEncoderV2, xLSTMEncoderV3


class TopologyLiteXTrackGAT(nn.Module):
    def __init__(self, config: Optional[dict] = None):
        super().__init__()
        config = config or {}
        self.input_dim = config.get("input_dim", 2)
        self.neighbor_input_dim = config.get("neighbor_input_dim", 4)
        self.embedding_dim = config.get("embedding_dim", 32)
        self.encoder_hidden = config.get("encoder_hidden", 64)
        self.decoder_hidden = config.get("decoder_hidden", 128)
        self.encoder_type = config.get("encoder_type", "xlstm_v3")
        self.num_heads = config.get("num_heads", 2)
        self.dropout = config.get("dropout", 0.2)
        self.gat_dropout = config.get("gat_dropout", 0.1)
        self.num_modes = int(config.get("num_modes", 6))
        self.dt = config.get("dt", 0.04)
        self.ax_max = config.get("ax_max", 9.0)
        self.psi_dot_max = config.get("psi_dot_max", 1.244)
        self.speed_max = config.get("speed_max", 20.0)
        self.use_map_signal = bool(config.get("use_map_signal", True))
        self.enable_behavior_conditioning = bool(config.get("enable_behavior_conditioning", False))
        self.decoder_type = config.get("decoder_type", "lstm")
        self.topology_output_dim = int(config.get("topology_output_dim", 24))
        self.topology_k_waypoints = int(config.get("topology_k_waypoints", 8))
        self.topology_k_hard_segments = int(config.get("topology_k_hard_segments", 8))
        self.topology_k_polygons = int(config.get("topology_k_polygons", 3))
        self.local_scene_radius = float(config.get("local_scene_radius", 20.0))

        self.embedding = InputEmbedding(self.input_dim, self.embedding_dim)
        self.neighbor_embedding = InputEmbedding(self.neighbor_input_dim, self.embedding_dim)
        self.encoder = self._build_encoder()
        self.encoder_dropout = nn.Dropout(self.dropout)

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
        self.map_token_encoder = None
        if self.use_map_signal:
            self.map_token_encoder = LocalMapTokenEncoder(
                waypoint_dim=8,
                hard_dim=8,
                polygon_dim=8,
                hidden_dim=config.get("topology_hidden_dim", 32),
                token_dim=self.context_dim,
                dropout=self.dropout,
            )
        self.scene_gat = HeteroSceneGAT(
            hidden_dim=self.context_dim,
            num_heads=self.num_heads,
            dropout=self.gat_dropout,
        )

        if self.enable_behavior_conditioning:
            self.behavior_query_dim = int(config.get("behavior_query_dim", self.context_dim // 2))
            self.coarse_control_points = int(config.get("coarse_control_points", 6))
            self.behavior_queries = nn.Parameter(torch.randn(self.num_modes, self.behavior_query_dim) * 0.02)
            self.behavior_context_proj = nn.Sequential(
                nn.Linear(self.context_dim + self.behavior_query_dim, self.context_dim),
                nn.LayerNorm(self.context_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.context_dim, self.context_dim),
            )
            self.coarse_control_head = nn.Sequential(
                nn.Linear(self.context_dim, self.decoder_hidden),
                nn.LayerNorm(self.decoder_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.decoder_hidden, self.coarse_control_points * 2),
            )
            self.mode_score_head = nn.Sequential(
                nn.Linear(self.context_dim, self.decoder_hidden),
                nn.LayerNorm(self.decoder_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.decoder_hidden, 1),
            )
            self.mode_queries = None
            self.mode_classifier = None
        else:
            self.behavior_query_dim = 0
            self.coarse_control_points = 0
            self.behavior_queries = None
            self.behavior_context_proj = None
            self.coarse_control_head = None
            self.mode_score_head = None
            self.mode_queries = nn.Parameter(torch.randn(self.num_modes, self.context_dim) * 0.02)
            self.mode_classifier = nn.Sequential(
                nn.Linear(self.context_dim, self.context_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(self.dropout),
                nn.Linear(self.context_dim, self.num_modes),
            )

        self.decoder = TopologyLiteDecoder(
            context_dim=self.context_dim,
            hidden_dim=self.decoder_hidden,
            dt=self.dt,
            ax_max=self.ax_max,
            psi_dot_max=self.psi_dot_max,
            speed_max=self.speed_max,
            dropout=self.dropout,
            safety_margin=config.get("safety_margin", 1.5),
            detach_map_queries=config.get("detach_map_queries", True),
            use_map_signal=self.use_map_signal,
            topology_hidden_dim=config.get("topology_hidden_dim", 32),
            topology_output_dim=config.get("topology_output_dim", 24),
            topology_k_waypoints=config.get("topology_k_waypoints", 8),
            topology_k_hard_segments=config.get("topology_k_hard_segments", 8),
            topology_k_polygons=config.get("topology_k_polygons", 3),
            topology_refresh_steps=config.get("topology_refresh_steps", 5),
            use_coarse_motion=self.enable_behavior_conditioning,
            local_map_radius=config.get("decoder_local_map_radius", self.local_scene_radius),
            decoder_type=self.decoder_type,
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

    @staticmethod
    def _interpolate_controls(control_keyframes: torch.Tensor, pred_len: int) -> torch.Tensor:
        batch_size, num_modes, num_points, ctrl_dim = control_keyframes.shape
        if num_points <= 1:
            return control_keyframes.expand(batch_size, num_modes, pred_len, ctrl_dim)
        flat = control_keyframes.reshape(batch_size * num_modes, num_points, ctrl_dim).transpose(1, 2)
        interp = F.interpolate(flat, size=pred_len, mode="linear", align_corners=True)
        return interp.transpose(1, 2).reshape(batch_size, num_modes, pred_len, ctrl_dim)

    def _build_state_features(
        self,
        init_state: torch.Tensor,
        slot_polygon_vertices: Optional[torch.Tensor],
        slot_polygon_vertex_mask: Optional[torch.Tensor],
        slot_polygon_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position = init_state[:, :2]
        in_slot = torch.zeros(position.shape[0], device=position.device, dtype=position.dtype)
        if (
            slot_polygon_vertices is not None
            and slot_polygon_vertex_mask is not None
            and slot_polygon_mask is not None
            and slot_polygon_vertices.shape[1] > 0
        ):
            _, _, inside = min_signed_distance_to_polygons(
                position,
                slot_polygon_vertices,
                slot_polygon_vertex_mask,
                slot_polygon_mask,
            )
            in_slot = inside.float()
        state_features = torch.stack(
            [
                init_state[:, 2] / max(float(self.speed_max), 1.0),
                torch.cos(init_state[:, 3]),
                torch.sin(init_state[:, 3]),
                in_slot,
            ],
            dim=-1,
        )
        return state_features, in_slot

    def _build_local_map_tokens(
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

    def _build_scene_context(
        self,
        h_target: torch.Tensor,
        h_neighbors: torch.Tensor,
        neighbor_trajs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        init_state: torch.Tensor,
        slot_polygon_vertices: Optional[torch.Tensor],
        slot_polygon_vertex_mask: Optional[torch.Tensor],
        slot_polygon_mask: Optional[torch.Tensor],
        hard_polygon_vertices: torch.Tensor,
        hard_polygon_vertex_mask: torch.Tensor,
        hard_polygon_mask: torch.Tensor,
        waypoint_segments: torch.Tensor,
        waypoint_segment_mask: torch.Tensor,
        hard_segments: torch.Tensor,
        hard_segment_mask: torch.Tensor,
        map_meta: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        position = init_state[:, :2]
        state_features, in_slot = self._build_state_features(
            init_state=init_state,
            slot_polygon_vertices=slot_polygon_vertices,
            slot_polygon_vertex_mask=slot_polygon_vertex_mask,
            slot_polygon_mask=slot_polygon_mask,
        )
        ego_token = self.ego_token_proj(torch.cat([h_target, state_features], dim=-1))

        neighbor_current = neighbor_trajs[:, :, -1, :]
        neighbor_rel_pos = neighbor_current[:, :, :2] - position.unsqueeze(1)
        neighbor_distance = torch.norm(neighbor_rel_pos, dim=-1)
        radius_scale = max(self.local_scene_radius, 1.0)
        neighbor_feature_tail = torch.cat([neighbor_rel_pos / radius_scale, neighbor_current[:, :, 2:4]], dim=-1)
        obstacle_tokens = self.obstacle_token_proj(torch.cat([h_neighbors, neighbor_feature_tail], dim=-1))
        obstacle_mask = neighbor_mask.bool() & (neighbor_distance <= self.local_scene_radius)
        obstacle_tokens = obstacle_tokens.masked_fill(~obstacle_mask.unsqueeze(-1), 0.0)

        map_tokens = self._build_local_map_tokens(
            position=position,
            hard_polygon_vertices=hard_polygon_vertices,
            hard_polygon_vertex_mask=hard_polygon_vertex_mask,
            hard_polygon_mask=hard_polygon_mask,
            waypoint_segments=waypoint_segments,
            waypoint_segment_mask=waypoint_segment_mask,
            hard_segments=hard_segments,
            hard_segment_mask=hard_segment_mask,
            map_meta=map_meta,
        )
        scene_context, graph_aux = self.scene_gat(
            ego_token=ego_token,
            obstacle_tokens=obstacle_tokens,
            obstacle_mask=obstacle_mask,
            hard_map_tokens=map_tokens["hard_tokens"],
            hard_map_mask=map_tokens["hard_mask"],
            soft_map_tokens=map_tokens["soft_tokens"],
            soft_map_mask=map_tokens["soft_mask"],
        )
        return scene_context, {
            "scene_in_slot": in_slot,
            "scene_attention": graph_aux["scene_attention"],
            "scene_node_count": graph_aux["scene_node_count"],
            "scene_obstacle_mask": obstacle_mask,
            "soft_token_mask": map_tokens["soft_mask"],
            "hard_token_mask": map_tokens["hard_mask"],
        }

    def _build_behavior_modes(
        self,
        scene_context: torch.Tensor,
        pred_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = scene_context.shape[0]
        behavior_queries = self.behavior_queries.unsqueeze(0).expand(batch_size, -1, -1)
        scene_expand = scene_context.unsqueeze(1).expand(-1, self.num_modes, -1)
        behavior_context = self.behavior_context_proj(torch.cat([scene_expand, behavior_queries], dim=-1))
        coarse_keyframes = self.coarse_control_head(behavior_context).view(batch_size, self.num_modes, self.coarse_control_points, 2)
        coarse_motion = self._interpolate_controls(coarse_keyframes, pred_len)
        mode_logits = self.mode_score_head(behavior_context).squeeze(-1)
        aux_outputs = {
            "behavior_queries": behavior_queries,
            "behavior_keyframes": coarse_keyframes,
            "coarse_motion": coarse_motion,
            "mode_valid_mask": torch.ones(batch_size, self.num_modes, device=scene_context.device, dtype=torch.bool),
        }
        return behavior_context, mode_logits, coarse_motion, aux_outputs

    def forward(
        self,
        obs_motion: torch.Tensor,
        init_state: torch.Tensor,
        neighbor_trajs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        pred_len: int,
        hard_polygon_vertices: torch.Tensor,
        hard_polygon_vertex_mask: torch.Tensor,
        hard_polygon_mask: torch.Tensor,
        waypoint_segments: torch.Tensor,
        waypoint_segment_mask: torch.Tensor,
        hard_segments: torch.Tensor,
        hard_segment_mask: torch.Tensor,
        map_meta: torch.Tensor,
        agent_size: torch.Tensor,
        slot_polygon_vertices: Optional[torch.Tensor] = None,
        slot_polygon_vertex_mask: Optional[torch.Tensor] = None,
        slot_polygon_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_neighbors, obs_len, _ = neighbor_trajs.shape
        target_emb = self.embedding(obs_motion)
        h_target = self.encoder_dropout(self.encoder(target_emb))

        neighbor_flat = neighbor_trajs.reshape(batch_size * num_neighbors, obs_len, self.neighbor_input_dim)
        neighbor_emb = self.neighbor_embedding(neighbor_flat)
        h_neighbors = self.encoder(neighbor_emb).reshape(batch_size, num_neighbors, self.encoder_hidden)
        h_neighbors = self.encoder_dropout(h_neighbors)

        scene_context, scene_aux = self._build_scene_context(
            h_target=h_target,
            h_neighbors=h_neighbors,
            neighbor_trajs=neighbor_trajs,
            neighbor_mask=neighbor_mask,
            init_state=init_state,
            slot_polygon_vertices=slot_polygon_vertices,
            slot_polygon_vertex_mask=slot_polygon_vertex_mask,
            slot_polygon_mask=slot_polygon_mask,
            hard_polygon_vertices=hard_polygon_vertices,
            hard_polygon_vertex_mask=hard_polygon_vertex_mask,
            hard_polygon_mask=hard_polygon_mask,
            waypoint_segments=waypoint_segments,
            waypoint_segment_mask=waypoint_segment_mask,
            hard_segments=hard_segments,
            hard_segment_mask=hard_segment_mask,
            map_meta=map_meta,
        )

        coarse_motion = None
        if self.enable_behavior_conditioning:
            mode_context, mode_logits, coarse_motion, aux_outputs = self._build_behavior_modes(
                scene_context=scene_context,
                pred_len=pred_len,
            )
        else:
            mode_context = scene_context.unsqueeze(1) + self.mode_queries.unsqueeze(0)
            mode_logits = self.mode_classifier(scene_context)
            aux_outputs = {"mode_valid_mask": torch.ones(batch_size, self.num_modes, device=scene_context.device, dtype=torch.bool)}

        aux_outputs.update(scene_aux)

        last_motion = obs_motion[:, -1, :]
        flat_context = mode_context.reshape(batch_size * self.num_modes, self.context_dim)
        flat_init_state = self._repeat_by_mode(init_state, self.num_modes)
        flat_last_motion = self._repeat_by_mode(last_motion, self.num_modes)
        flat_agent_size = self._repeat_by_mode(agent_size, self.num_modes)
        flat_hard_polygon_vertices = self._repeat_by_mode(hard_polygon_vertices, self.num_modes)
        flat_hard_polygon_vertex_mask = self._repeat_by_mode(hard_polygon_vertex_mask, self.num_modes)
        flat_hard_polygon_mask = self._repeat_by_mode(hard_polygon_mask, self.num_modes)
        flat_waypoint_segments = self._repeat_by_mode(waypoint_segments, self.num_modes)
        flat_waypoint_segment_mask = self._repeat_by_mode(waypoint_segment_mask, self.num_modes)
        flat_hard_segments = self._repeat_by_mode(hard_segments, self.num_modes)
        flat_hard_segment_mask = self._repeat_by_mode(hard_segment_mask, self.num_modes)
        flat_map_meta = self._repeat_by_mode(map_meta, self.num_modes)
        flat_coarse_motion = None
        if coarse_motion is not None:
            flat_coarse_motion = coarse_motion.reshape(batch_size * self.num_modes, pred_len, 2)

        pred_pos_flat, pred_motion_flat, _ = self.decoder(
            base_context=flat_context,
            pred_len=pred_len,
            init_state=flat_init_state,
            last_motion=flat_last_motion,
            coarse_motion=flat_coarse_motion,
            agent_size=flat_agent_size,
            hard_polygon_vertices=flat_hard_polygon_vertices,
            hard_polygon_vertex_mask=flat_hard_polygon_vertex_mask,
            hard_polygon_mask=flat_hard_polygon_mask,
            waypoint_segments=flat_waypoint_segments,
            waypoint_segment_mask=flat_waypoint_segment_mask,
            hard_segments=flat_hard_segments,
            hard_segment_mask=flat_hard_segment_mask,
            map_meta=flat_map_meta,
        )
        pred_pos = pred_pos_flat.reshape(batch_size, self.num_modes, pred_len, 2)
        pred_motion = pred_motion_flat.reshape(batch_size, self.num_modes, pred_len, 2)
        return pred_pos, pred_motion, mode_logits, aux_outputs
