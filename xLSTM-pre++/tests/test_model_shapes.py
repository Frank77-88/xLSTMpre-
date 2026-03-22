"""Smoke tests for the five formal xLSTM-pre++ variants."""

from pathlib import Path

import numpy as np
import torch

from xlstm_prepp.data import build_map_adapter, load_config, resolve_path
from xlstm_prepp.data.preprocessing import compute_motion_params
from xlstm_prepp.models.kinematic_layer import KinematicLayer
from xlstm_prepp.runtime import create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = [
    "GAT-xLSTM-K-I++.yaml",
    "GAT-xLSTM-K++.yaml",
    "GAT-xLSTM++.yaml",
    "xLSTM++.yaml",
    "LSTM++.yaml",
]


def _load_config(name: str):
    return load_config(resolve_path(f"configs/{name}", str(PROJECT_ROOT)))


def _expand_token_bank(token_bank, batch_size: int):
    def repeat(tensor):
        return tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.ndim)).contiguous()

    return {
        "map_meta": repeat(token_bank.map_meta),
        "slot_polygon_vertices": repeat(token_bank.slot_polygon_vertices),
        "slot_polygon_vertex_mask": repeat(token_bank.slot_polygon_vertex_mask),
        "slot_polygon_mask": repeat(token_bank.slot_polygon_mask),
        "hard_polygon_vertices": repeat(token_bank.hard_polygon_vertices),
        "hard_polygon_vertex_mask": repeat(token_bank.hard_polygon_vertex_mask),
        "hard_polygon_mask": repeat(token_bank.hard_polygon_mask),
        "waypoint_segments": repeat(token_bank.waypoint_segments),
        "waypoint_segment_mask": repeat(token_bank.waypoint_segment_mask),
        "hard_segments": repeat(token_bank.hard_segments),
        "hard_segment_mask": repeat(token_bank.hard_segment_mask),
    }


def test_all_configs_loadable():
    for name in CONFIG_NAMES:
        config = _load_config(name)
        model = create_model(config)
        assert model is not None, name


def test_behavior_k_forward_shape():
    config = _load_config("GAT-xLSTM-K-I++.yaml")
    model = create_model(config)
    adapter = build_map_adapter(config)
    token_bank = adapter.build_token_bank()

    batch_size = 2
    pred_len = config["data"]["pred_len"]
    obs_len = config["data"]["obs_len"]
    num_neighbors = config["data"]["num_neighbors"]
    num_modes = config["model"]["num_modes"]
    map_batch = _expand_token_bank(token_bank, batch_size)

    init_state = torch.randn(batch_size, 4)
    init_state[:, 2] = init_state[:, 2].abs() + 1.0
    pred_pos, pred_motion, mode_logits, aux_outputs = model(
        torch.randn(batch_size, obs_len, config["model"]["input_dim"]),
        init_state,
        torch.randn(batch_size, num_neighbors, obs_len, config["model"]["neighbor_input_dim"]),
        torch.ones(batch_size, num_neighbors),
        pred_len,
        map_batch["hard_polygon_vertices"],
        map_batch["hard_polygon_vertex_mask"],
        map_batch["hard_polygon_mask"],
        map_batch["waypoint_segments"],
        map_batch["waypoint_segment_mask"],
        map_batch["hard_segments"],
        map_batch["hard_segment_mask"],
        map_batch["map_meta"],
        torch.tensor([[4.8, 2.0], [5.2, 2.1]], dtype=torch.float32),
        map_batch["slot_polygon_vertices"],
        map_batch["slot_polygon_vertex_mask"],
        map_batch["slot_polygon_mask"],
    )
    assert pred_pos.shape == (batch_size, num_modes, pred_len, 2)
    assert pred_motion.shape == (batch_size, num_modes, pred_len, 2)
    assert mode_logits.shape == (batch_size, num_modes)
    assert aux_outputs["behavior_queries"].shape[:2] == (batch_size, num_modes)
    assert aux_outputs["behavior_keyframes"].shape == (batch_size, num_modes, config["model"]["coarse_control_points"], 2)
    assert aux_outputs["coarse_motion"].shape == (batch_size, num_modes, pred_len, 2)


def test_behavior_kf_forward_shape():
    config = _load_config("GAT-xLSTM-K-F++.yaml")
    model = create_model(config)
    adapter = build_map_adapter(config)
    token_bank = adapter.build_token_bank()

    batch_size = 2
    pred_len = config["data"]["pred_len"]
    obs_len = config["data"]["obs_len"]
    num_neighbors = config["data"]["num_neighbors"]
    num_modes = config["model"]["num_modes"]
    map_batch = _expand_token_bank(token_bank, batch_size)

    init_state = torch.randn(batch_size, 4)
    init_state[:, 2] = init_state[:, 2].abs() + 1.0
    pred_pos, pred_motion, mode_logits, aux_outputs = model(
        torch.randn(batch_size, obs_len, config["model"]["input_dim"]),
        init_state,
        torch.randn(batch_size, num_neighbors, obs_len, config["model"]["neighbor_input_dim"]),
        torch.ones(batch_size, num_neighbors),
        pred_len,
        map_batch["hard_polygon_vertices"],
        map_batch["hard_polygon_vertex_mask"],
        map_batch["hard_polygon_mask"],
        map_batch["waypoint_segments"],
        map_batch["waypoint_segment_mask"],
        map_batch["hard_segments"],
        map_batch["hard_segment_mask"],
        map_batch["map_meta"],
        torch.tensor([[4.8, 2.0], [5.2, 2.1]], dtype=torch.float32),
        map_batch["slot_polygon_vertices"],
        map_batch["slot_polygon_vertex_mask"],
        map_batch["slot_polygon_mask"],
    )
    assert pred_pos.shape == (batch_size, num_modes, pred_len, 2)
    assert pred_motion.shape == (batch_size, num_modes, pred_len, 2)
    assert mode_logits.shape == (batch_size, num_modes)
    assert aux_outputs["behavior_queries"].shape[:2] == (batch_size, num_modes)


def test_gat_xlstm_forward_shape():
    config = _load_config("GAT-xLSTM++.yaml")
    model = create_model(config)
    adapter = build_map_adapter(config)
    token_bank = adapter.build_token_bank()
    batch_size = 2
    pred_len = config["data"]["pred_len"]
    obs_len = config["data"]["obs_len"]
    num_neighbors = config["data"]["num_neighbors"]
    num_modes = config["model"]["num_modes"]
    map_batch = _expand_token_bank(token_bank, batch_size)
    pred_pos, pred_motion, mode_logits = model(
        torch.randn(batch_size, obs_len, config["model"]["input_dim"]),
        pred_len,
        torch.randn(batch_size, num_neighbors, obs_len, config["model"]["neighbor_input_dim"]),
        torch.ones(batch_size, num_neighbors),
        map_batch["hard_polygon_vertices"],
        map_batch["hard_polygon_vertex_mask"],
        map_batch["hard_polygon_mask"],
        map_batch["waypoint_segments"],
        map_batch["waypoint_segment_mask"],
        map_batch["hard_segments"],
        map_batch["hard_segment_mask"],
        map_batch["map_meta"],
    )
    assert pred_pos.shape == (batch_size, num_modes, pred_len, 2)
    assert pred_motion is None
    assert mode_logits.shape == (batch_size, num_modes)

def test_gat_k_forward_shape():
    config = _load_config("GAT-xLSTM-K++.yaml")
    model = create_model(config)
    adapter = build_map_adapter(config)
    token_bank = adapter.build_token_bank()
    batch_size = 2
    pred_len = config["data"]["pred_len"]
    obs_len = config["data"]["obs_len"]
    num_neighbors = config["data"]["num_neighbors"]
    num_modes = config["model"]["num_modes"]
    map_batch = _expand_token_bank(token_bank, batch_size)
    pred_pos, pred_motion, mode_logits, aux_outputs = model(
        torch.randn(batch_size, obs_len, config["model"]["input_dim"]),
        torch.randn(batch_size, 4),
        torch.randn(batch_size, num_neighbors, obs_len, config["model"]["neighbor_input_dim"]),
        torch.ones(batch_size, num_neighbors),
        pred_len,
        map_batch["hard_polygon_vertices"],
        map_batch["hard_polygon_vertex_mask"],
        map_batch["hard_polygon_mask"],
        map_batch["waypoint_segments"],
        map_batch["waypoint_segment_mask"],
        map_batch["hard_segments"],
        map_batch["hard_segment_mask"],
        map_batch["map_meta"],
        torch.tensor([[4.8, 2.0], [5.2, 2.1]], dtype=torch.float32),
        map_batch["slot_polygon_vertices"],
        map_batch["slot_polygon_vertex_mask"],
        map_batch["slot_polygon_mask"],
    )
    assert pred_pos.shape == (batch_size, num_modes, pred_len, 2)
    assert pred_motion.shape == (batch_size, num_modes, pred_len, 2)
    assert mode_logits.shape == (batch_size, num_modes)
    assert aux_outputs["scene_node_count"].shape == (batch_size,)


def test_xtraj_v3_forward_shape():
    config = _load_config("xLSTM++.yaml")
    model = create_model(config)
    adapter = build_map_adapter(config)
    token_bank = adapter.build_token_bank()
    batch_size = 2
    pred_len = config["data"]["pred_len"]
    obs_len = config["data"]["obs_len"]
    num_modes = config["model"]["num_modes"]
    map_batch = _expand_token_bank(token_bank, batch_size)
    pred_pos, pred_motion, mode_logits = model(
        torch.randn(batch_size, obs_len, config["model"]["input_dim"]),
        pred_len,
        hard_polygon_vertices=map_batch["hard_polygon_vertices"],
        hard_polygon_vertex_mask=map_batch["hard_polygon_vertex_mask"],
        hard_polygon_mask=map_batch["hard_polygon_mask"],
        waypoint_segments=map_batch["waypoint_segments"],
        waypoint_segment_mask=map_batch["waypoint_segment_mask"],
        hard_segments=map_batch["hard_segments"],
        hard_segment_mask=map_batch["hard_segment_mask"],
        map_meta=map_batch["map_meta"],
    )
    assert pred_pos.shape == (batch_size, num_modes, pred_len, 2)
    assert pred_motion is None
    assert mode_logits.shape == (batch_size, num_modes)


def test_compute_motion_params_supports_reverse_speed():
    trajectory = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [-2.0, 0.0, 0.0, 1.0],
            [-3.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    params = compute_motion_params(trajectory, dt=1.0)
    assert np.allclose(params["v"], np.array([-1.0, -1.0, -1.0], dtype=np.float32))
    assert np.allclose(params["ax"], np.zeros(2, dtype=np.float32))


def test_kinematic_layer_allows_negative_speed():
    layer = KinematicLayer(dt=1.0, ax_max=9.0, psi_dot_max=1.244, speed_max=10.0)
    init_state = torch.tensor([[0.0, 0.0, -2.0, 0.0]], dtype=torch.float32)
    motion = torch.zeros(1, 3, 2, dtype=torch.float32)
    positions = layer.get_positions(motion, init_state)
    expected = torch.tensor([[[-2.0, 0.0], [-4.0, 0.0], [-6.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(positions, expected)
