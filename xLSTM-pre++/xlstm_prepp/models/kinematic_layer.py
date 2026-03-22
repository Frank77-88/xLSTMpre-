"""物理运动学层。"""

import torch
import torch.nn as nn


class KinematicLayer(nn.Module):
    """将预测的运动参数 (ax, psi_dot) 转换为状态 / 位置轨迹。"""

    def __init__(self, dt: float = 0.04, ax_max: float = 9.0, psi_dot_max: float = 1.244, speed_max: float = 20.0):
        super().__init__()
        self.dt = dt
        self.ax_max = ax_max
        self.psi_dot_max = psi_dot_max
        self.speed_max = abs(float(speed_max))
        self.dt2 = dt * dt / 2.0

    def step(self, state: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        ax = torch.clamp(motion[:, 0], -self.ax_max, self.ax_max)
        psi_dot = torch.clamp(motion[:, 1], -self.psi_dot_max, self.psi_dot_max)

        x = state[:, 0]
        y = state[:, 1]
        speed = state[:, 2]
        psi = state[:, 3]
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        next_x = x + speed * cos_psi * self.dt + (ax * cos_psi - psi_dot * speed * sin_psi) * self.dt2
        next_y = y + speed * sin_psi * self.dt + (ax * sin_psi + psi_dot * speed * cos_psi) * self.dt2
        next_speed = torch.clamp(speed + ax * self.dt, min=-self.speed_max, max=self.speed_max)
        next_psi = psi + psi_dot * self.dt
        return torch.stack([next_x, next_y, next_speed, next_psi], dim=-1)

    def forward(self, motion_params: torch.Tensor, init_state: torch.Tensor) -> torch.Tensor:
        batch_size, pred_len, _ = motion_params.shape
        current_state = init_state
        states = []
        for step_index in range(pred_len):
            current_state = self.step(current_state, motion_params[:, step_index, :])
            states.append(current_state)
        return torch.stack(states, dim=1).view(batch_size, pred_len, 4)

    def get_positions(self, motion_params: torch.Tensor, init_state: torch.Tensor) -> torch.Tensor:
        states = self.forward(motion_params, init_state)
        return states[:, :, :2]
