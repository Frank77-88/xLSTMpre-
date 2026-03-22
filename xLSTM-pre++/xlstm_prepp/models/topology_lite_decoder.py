"""Kinematic decoder with lightweight local topology context."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from xlstm_prepp.map_geometry import (
    build_local_topology_features,
    compute_point_map_bundle,
    select_local_map_subset,
)

from .kinematic_layer import KinematicLayer
from .mlstm import mLSTMCell
from .slstm import sLSTMCell
from .topology_lite_encoder import LocalTopologyLiteEncoder


class DecoderxLSTMBlock(nn.Module):
    """Step-wise xLSTM block used by the experimental decoder variant."""

    def __init__(self, input_dim: int, hidden_dim: int, base_context_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_context_dim = hidden_dim if base_context_dim is None else int(base_context_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self.slstm_norm = nn.LayerNorm(hidden_dim)
        self.slstm_cell = sLSTMCell(hidden_dim, hidden_dim)
        self.slstm_dropout = nn.Dropout(dropout)

        self.mlstm_norm = nn.LayerNorm(hidden_dim)
        self.mlstm_cell = mLSTMCell(hidden_dim, hidden_dim)
        self.mlstm_dropout = nn.Dropout(dropout)

        self.output_norm = nn.LayerNorm(hidden_dim)

        self.init_s_h = nn.Linear(self.base_context_dim, hidden_dim)
        self.init_s_c = nn.Linear(self.base_context_dim, hidden_dim)
        self.init_m_diag = nn.Linear(self.base_context_dim, hidden_dim)
        self.init_m_n = nn.Linear(self.base_context_dim, hidden_dim)

    def init_state(self, base_context: torch.Tensor):
        batch_size = base_context.shape[0]
        device = base_context.device
        dtype = base_context.dtype

        s_h = torch.tanh(self.init_s_h(base_context))
        s_c = torch.tanh(self.init_s_c(base_context))
        s_n = torch.ones(batch_size, self.hidden_dim, device=device, dtype=dtype)

        m_diag = torch.tanh(self.init_m_diag(base_context))
        m_C = torch.diag_embed(m_diag)
        m_n = torch.tanh(self.init_m_n(base_context))
        return s_h, s_c, s_n, m_C, m_n

    def forward(self, x_t: torch.Tensor, state):
        s_h, s_c, s_n, m_C, m_n = state

        x_t = self.input_proj(x_t)

        s_in = self.slstm_norm(x_t)
        s_h, s_c, s_n = self.slstm_cell(s_in, s_h, s_c, s_n)
        h_res = x_t + self.slstm_dropout(s_h)

        m_in = self.mlstm_norm(h_res)
        m_h, m_C, m_n = self.mlstm_cell(m_in, m_C, m_n)
        h_out = self.output_norm(h_res + self.mlstm_dropout(m_h))
        return h_out, (s_h, s_c, s_n, m_C, m_n)


class TopologyLiteDecoder(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        dt: float,
        ax_max: float,
        psi_dot_max: float,
        speed_max: float,
        dropout: float = 0.1,
        safety_margin: float = 1.5,
        detach_map_queries: bool = True,
        use_map_signal: bool = True,
        topology_hidden_dim: int = 32,
        topology_output_dim: int = 24,
        topology_k_waypoints: int = 8,
        topology_k_hard_segments: int = 8,
        topology_k_polygons: int = 3,
        topology_refresh_steps: int = 5,
        use_coarse_motion: bool = False,
        local_map_radius: float | None = None,
        decoder_type: str = "lstm",
    ):
        super().__init__()
        self.ax_max = ax_max
        self.psi_dot_max = psi_dot_max
        self.speed_max = speed_max
        self.safety_margin = safety_margin
        self.detach_map_queries = detach_map_queries
        self.use_map_signal = use_map_signal
        self.topology_k_waypoints = topology_k_waypoints
        self.topology_k_hard_segments = topology_k_hard_segments
        self.topology_k_polygons = topology_k_polygons
        self.topology_refresh_steps = max(int(topology_refresh_steps), 1)
        self.use_coarse_motion = bool(use_coarse_motion)
        self.local_map_radius = None if local_map_radius is None else float(local_map_radius)
        self.decoder_type = str(decoder_type).lower()

        self.hard_signal_dim = 13 if self.use_map_signal else 0
        self.topology_context_dim = topology_output_dim if self.use_map_signal else 0
        self.coarse_motion_dim = 2 if self.use_coarse_motion else 0
        decoder_input_dim = context_dim + 2 + 2 + 2 + self.coarse_motion_dim + self.hard_signal_dim + self.topology_context_dim

        if self.decoder_type == "xlstm_block":
            self.init_h = None
            self.init_c = None
            self.cell = None
            self.xlstm_block = DecoderxLSTMBlock(
                decoder_input_dim,
                hidden_dim,
                base_context_dim=context_dim,
                dropout=dropout,
            )
        else:
            self.init_h = nn.Linear(context_dim, hidden_dim)
            self.init_c = nn.Linear(context_dim, hidden_dim)
            self.cell = nn.LSTMCell(decoder_input_dim, hidden_dim)
            self.xlstm_block = None
        self.motion_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.kinematics = KinematicLayer(dt=dt, ax_max=ax_max, psi_dot_max=psi_dot_max, speed_max=speed_max)
        self.topology_encoder = None
        if self.use_map_signal:
            self.topology_encoder = LocalTopologyLiteEncoder(
                waypoint_dim=8,
                hard_dim=8,
                polygon_dim=8,
                hidden_dim=topology_hidden_dim,
                output_dim=topology_output_dim,
                dropout=dropout,
            )

    @staticmethod
    def _heading(psi: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)

    def _compute_bundle(
        self,
        current_state: torch.Tensor,
        agent_size: torch.Tensor,
        hard_polygon_vertices: torch.Tensor,
        hard_polygon_vertex_mask: torch.Tensor,
        hard_polygon_mask: torch.Tensor,
        waypoint_segments: torch.Tensor,
        waypoint_segment_mask: torch.Tensor,
        hard_segments: torch.Tensor,
        hard_segment_mask: torch.Tensor,
        map_meta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        position = current_state[:, :2]
        heading = current_state[:, 3]
        if self.use_map_signal:
            local_map = select_local_map_subset(
                position=position.detach() if self.detach_map_queries else position,
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
                local_radius=self.local_map_radius,
            )
            hard_polygon_vertices = local_map["hard_polygon_vertices"]
            hard_polygon_vertex_mask = local_map["hard_polygon_vertex_mask"]
            hard_polygon_mask = local_map["hard_polygon_mask"]
            waypoint_segments = local_map["waypoint_segments"]
            waypoint_segment_mask = local_map["waypoint_mask"]
            hard_segments = local_map["hard_segments"]
            hard_segment_mask = local_map["hard_mask"]
        if self.detach_map_queries:
            position = position.detach()
            heading = heading.detach()
            with torch.no_grad():
                return compute_point_map_bundle(
                    position=position,
                    heading=heading,
                    agent_size=agent_size,
                    hard_polygon_vertices=hard_polygon_vertices,
                    hard_polygon_vertex_mask=hard_polygon_vertex_mask,
                    hard_polygon_mask=hard_polygon_mask,
                    waypoint_segments=waypoint_segments,
                    waypoint_segment_mask=waypoint_segment_mask,
                    hard_segments=hard_segments,
                    hard_segment_mask=hard_segment_mask,
                    map_meta=map_meta,
                    safety_margin=self.safety_margin,
                )
        return compute_point_map_bundle(
            position=position,
            heading=heading,
            agent_size=agent_size,
            hard_polygon_vertices=hard_polygon_vertices,
            hard_polygon_vertex_mask=hard_polygon_vertex_mask,
            hard_polygon_mask=hard_polygon_mask,
            waypoint_segments=waypoint_segments,
            waypoint_segment_mask=waypoint_segment_mask,
            hard_segments=hard_segments,
            hard_segment_mask=hard_segment_mask,
            map_meta=map_meta,
            safety_margin=self.safety_margin,
        )

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

        query_position = position.detach() if self.detach_map_queries else position
        if self.detach_map_queries:
            with torch.no_grad():
                local_map = select_local_map_subset(
                    position=query_position,
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
                    local_radius=self.local_map_radius,
                )
                topology_inputs = build_local_topology_features(query_position, local_map)
        else:
            local_map = select_local_map_subset(
                position=query_position,
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
                local_radius=self.local_map_radius,
            )
            topology_inputs = build_local_topology_features(query_position, local_map)

        return self.topology_encoder(
            waypoint_features=topology_inputs["waypoint_features"],
            waypoint_mask=topology_inputs["waypoint_mask"],
            hard_features=topology_inputs["hard_features"],
            hard_mask=topology_inputs["hard_mask"],
            polygon_features=topology_inputs["polygon_features"],
            polygon_mask=topology_inputs["polygon_mask"],
        )

    def _zero_hard_signal(self, current_state: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            current_state.shape[0],
            self.hard_signal_dim,
            device=current_state.device,
            dtype=current_state.dtype,
        )

    def _build_hard_signal(self, bundle: Dict[str, torch.Tensor], map_meta: torch.Tensor) -> torch.Tensor:
        map_scale = map_meta[:, 2:3].clamp_min(1.0)
        return torch.cat([bundle["signal"], bundle["hard_correction"] / map_scale], dim=-1)

    def forward(
        self,
        base_context: torch.Tensor,
        pred_len: int,
        init_state: torch.Tensor,
        last_motion: torch.Tensor,
        coarse_motion: torch.Tensor | None,
        agent_size: torch.Tensor,
        hard_polygon_vertices: torch.Tensor,
        hard_polygon_vertex_mask: torch.Tensor,
        hard_polygon_mask: torch.Tensor,
        waypoint_segments: torch.Tensor,
        waypoint_segment_mask: torch.Tensor,
        hard_segments: torch.Tensor,
        hard_segment_mask: torch.Tensor,
        map_meta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.decoder_type == "xlstm_block":
            recurrent_state = self.xlstm_block.init_state(base_context)
        else:
            hidden_state = torch.tanh(self.init_h(base_context))
            cell_state = torch.tanh(self.init_c(base_context))
        current_state = init_state
        prev_motion = last_motion

        pred_positions = []
        pred_motions = []
        need_geometry = self.use_map_signal
        topology_context = None

        for step_idx in range(pred_len):
            if need_geometry:
                bundle = self._compute_bundle(
                    current_state=current_state,
                    agent_size=agent_size,
                    hard_polygon_vertices=hard_polygon_vertices,
                    hard_polygon_vertex_mask=hard_polygon_vertex_mask,
                    hard_polygon_mask=hard_polygon_mask,
                    waypoint_segments=waypoint_segments,
                    waypoint_segment_mask=waypoint_segment_mask,
                    hard_segments=hard_segments,
                    hard_segment_mask=hard_segment_mask,
                    map_meta=map_meta,
                )
                hard_signal = self._build_hard_signal(bundle, map_meta)
            else:
                bundle = None
                hard_signal = self._zero_hard_signal(current_state)

            if self.use_map_signal:
                if topology_context is None or (step_idx % self.topology_refresh_steps == 0):
                    topology_context = self._compute_topology_context(
                        position=current_state[:, :2],
                        hard_polygon_vertices=hard_polygon_vertices,
                        hard_polygon_vertex_mask=hard_polygon_vertex_mask,
                        hard_polygon_mask=hard_polygon_mask,
                        waypoint_segments=waypoint_segments,
                        waypoint_segment_mask=waypoint_segment_mask,
                        hard_segments=hard_segments,
                        hard_segment_mask=hard_segment_mask,
                        map_meta=map_meta,
                    )
            else:
                topology_context = torch.zeros(current_state.shape[0], 0, device=current_state.device, dtype=current_state.dtype)

            heading_vec = self._heading(current_state[:, 3])
            if coarse_motion is not None:
                coarse_motion_t = coarse_motion[:, step_idx, :]
            else:
                coarse_motion_t = torch.zeros(
                    current_state.shape[0],
                    self.coarse_motion_dim,
                    device=current_state.device,
                    dtype=current_state.dtype,
                )
            decoder_input = torch.cat(
                [
                    base_context,
                    current_state[:, 2:4],
                    prev_motion,
                    heading_vec,
                    coarse_motion_t,
                    hard_signal,
                    topology_context,
                ],
                dim=-1,
            )
            if self.decoder_type == "xlstm_block":
                hidden_state, recurrent_state = self.xlstm_block(decoder_input, recurrent_state)
            else:
                hidden_state, cell_state = self.cell(decoder_input, (hidden_state, cell_state))
            delta_motion = self.motion_head(hidden_state)
            motion = delta_motion + coarse_motion_t if coarse_motion is not None else delta_motion
            next_state = self.kinematics.step(current_state, motion)

            pred_positions.append(next_state[:, :2])
            pred_motions.append(motion)
            current_state = next_state
            prev_motion = motion

        return torch.stack(pred_positions, dim=1), torch.stack(pred_motions, dim=1), {}
