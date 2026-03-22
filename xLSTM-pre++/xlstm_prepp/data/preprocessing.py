"""Native preprocessing for xLSTM-pre++."""

from __future__ import annotations

from typing import Dict

import numpy as np


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_motion_params(trajectory: np.ndarray, dt: float = 0.04) -> Dict[str, np.ndarray]:
    if trajectory.shape[1] < 2:
        raise ValueError(f"轨迹维度错误: {trajectory.shape}")

    pos = trajectory[:, :2]
    num_steps = len(pos)
    if num_steps < 3:
        raise ValueError(f"轨迹长度不足: {num_steps} < 3")

    has_heading = trajectory.shape[1] >= 3
    has_speed = trajectory.shape[1] >= 4

    delta = pos[1:] - pos[:-1]
    velocity = delta / dt

    if has_speed:
        speed_mag = np.abs(trajectory[:-1, 3].copy())
    else:
        speed_mag = np.linalg.norm(velocity, axis=1)

    if has_heading:
        heading = trajectory[:-1, 2].copy()
    else:
        heading = np.arctan2(velocity[:, 1], velocity[:, 0])

    heading_vec = np.stack([np.cos(heading), np.sin(heading)], axis=1)
    projected_speed = np.sum(velocity * heading_vec, axis=1)
    speed_sign = np.sign(projected_speed)
    speed_sign[np.abs(projected_speed) < 1e-4] = 1.0
    speed = speed_mag * speed_sign
    speed[speed_mag < 1e-4] = 0.0

    dpsi = wrap_angle(heading[1:] - heading[:-1])
    psi_dot = dpsi / dt
    dv = speed[1:] - speed[:-1]
    ax = dv / dt

    return {
        "pos": pos,
        "v": speed,
        "psi": heading,
        "ax": ax,
        "psi_dot": psi_dot,
    }
