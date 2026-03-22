"""Runtime helpers for model creation."""

from __future__ import annotations

from typing import Dict

from xlstm_prepp.data import get_model_type
from xlstm_prepp.models import TopologyLiteXTrackGAT, XTrajMultiModalPredictor

SUPPORTED_MODEL_TYPES = {
    "topology_lite_xtrack_gat",
    "xtrack_gat_pp",
    "xtraj_gat_pp",
    "xtraj_pp",
}


def prepare_model_config(config: Dict) -> Dict:
    model_type = get_model_type(config)
    model_config = dict(config.get("model", {}))
    physics_config = config.get("physics", {})
    map_config = config.get("map", {})
    data_config = config.get("data", {})

    model_config.update(
        {
            "dt": data_config.get("dt", physics_config.get("dt", 0.04)),
            "ax_max": physics_config.get("ax_max", 9.0),
            "psi_dot_max": physics_config.get("psi_dot_max", 1.244),
            "speed_max": physics_config.get("speed_max", 20.0),
            "safety_margin": map_config.get("safety_margin", 1.5),
            "detach_map_queries": map_config.get("detach_map_queries", True),
            "use_map_signal": map_config.get("use_map_signal", True),
            "topology_hidden_dim": map_config.get("topology_hidden_dim", 32),
            "topology_output_dim": map_config.get("topology_output_dim", 24),
            "topology_k_waypoints": map_config.get("topology_k_waypoints", 8),
            "topology_k_hard_segments": map_config.get("topology_k_hard_segments", 8),
            "topology_k_polygons": map_config.get("topology_k_polygons", 3),
            "topology_refresh_steps": map_config.get("topology_refresh_steps", 5),
            "local_scene_radius": data_config.get("local_scene_radius", 20.0),
            "decoder_local_map_radius": map_config.get("decoder_local_map_radius", data_config.get("local_scene_radius", 20.0)),
            "enable_behavior_conditioning": model_config.get("enable_behavior_conditioning", False),
            "behavior_query_dim": model_config.get("behavior_query_dim", model_config.get("encoder_hidden", 64) // 2),
            "coarse_control_points": model_config.get("coarse_control_points", 6),
        }
    )

    if model_type == "xtraj_gat_pp":
        model_config["use_gat"] = True
    elif model_type == "xtraj_pp":
        model_config["use_gat"] = False

    return model_config


def create_model(config: Dict):
    model_type = get_model_type(config)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_config = prepare_model_config(config)
    if model_type in {"topology_lite_xtrack_gat", "xtrack_gat_pp"}:
        return TopologyLiteXTrackGAT(model_config)
    return XTrajMultiModalPredictor(model_config)
