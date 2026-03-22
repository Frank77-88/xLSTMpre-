"""xLSTM 混合编码器

组合 sLSTM + mLSTM 的混合架构:
- sLSTM: 局部时序特征提取，指数门控捕捉短期依赖
- mLSTM: 全局模式整合，矩阵记忆存储长程依赖

架构:
    Input [batch, seq_len, input_dim]
        ↓
    sLSTM Layer (局部特征)
        ↓ [batch, seq_len, hidden_dim]
    mLSTM Layer (全局整合)
        ↓ [batch, hidden_dim]
    Output
"""

import torch
import torch.nn as nn
from typing import Optional

from .slstm import sLSTMCell
from .mlstm import mLSTMCell


class xLSTMEncoder(nn.Module):
    """xLSTM 混合编码器
    
    sLSTM → mLSTM 的两层混合架构。
    
    Args:
        input_dim: 输入维度 (嵌入维度)
        hidden_dim: 隐藏状态维度 (默认 64)
        dropout: Dropout 概率 (默认 0.1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Layer 1: sLSTM (局部时序特征)
        self.slstm_cell = sLSTMCell(input_dim, hidden_dim)
        
        # 中间层归一化和 Dropout
        self.inter_norm = nn.LayerNorm(hidden_dim)
        self.inter_dropout = nn.Dropout(dropout)
        
        # Layer 2: mLSTM (全局模式整合)
        self.mlstm_cell = mLSTMCell(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_states: bool = False
    ) -> torch.Tensor:
        """编码输入序列
        
        Args:
            x: [batch, seq_len, input_dim] 输入序列
            return_all_states: 是否返回所有时刻的隐藏状态
        
        Returns:
            如果 return_all_states=False:
                [batch, hidden_dim] 最终隐藏状态
            如果 return_all_states=True:
                [batch, seq_len, hidden_dim] 所有时刻的隐藏状态
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # === sLSTM 初始化 ===
        h_s = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        c_s = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        n_s = torch.ones(batch, self.hidden_dim, device=device, dtype=dtype)
        
        # === mLSTM 初始化 ===
        C_m = torch.zeros(batch, self.hidden_dim, self.hidden_dim, device=device, dtype=dtype)
        n_m = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        
        all_h = []
        
        # 逐时间步处理
        for t in range(seq_len):
            # Layer 1: sLSTM
            h_s, c_s, n_s = self.slstm_cell(x[:, t], h_s, c_s, n_s)
            
            # 中间处理
            h_inter = self.inter_norm(h_s)
            h_inter = self.inter_dropout(h_inter)
            
            # Layer 2: mLSTM
            h_m, C_m, n_m = self.mlstm_cell(h_inter, C_m, n_m)
            
            if return_all_states:
                all_h.append(h_m)
        
        if return_all_states:
            return torch.stack(all_h, dim=1)
        else:
            return h_m


class xLSTMEncoderV2(nn.Module):
    """xLSTM 混合编码器 V2
    
    带残差连接的 sLSTM → mLSTM 架构。
    
    Args:
        input_dim: 输入维度 (嵌入维度)
        hidden_dim: 隐藏状态维度 (默认 64)
        dropout: Dropout 概率 (默认 0.1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影 (如果维度不匹配)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # Layer 1: sLSTM
        self.slstm_cell = sLSTMCell(hidden_dim, hidden_dim)
        self.slstm_norm = nn.LayerNorm(hidden_dim)
        self.slstm_dropout = nn.Dropout(dropout)
        
        # Layer 2: mLSTM
        self.mlstm_cell = mLSTMCell(hidden_dim, hidden_dim)
        self.mlstm_dropout = nn.Dropout(dropout)
        
        # 融合门控 (学习 sLSTM 和 mLSTM 输出的权重)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_states: bool = False
    ) -> torch.Tensor:
        """编码输入序列
        
        Args:
            x: [batch, seq_len, input_dim] 输入序列
            return_all_states: 是否返回所有时刻的隐藏状态
        
        Returns:
            如果 return_all_states=False:
                [batch, hidden_dim] 最终隐藏状态
            如果 return_all_states=True:
                [batch, seq_len, hidden_dim] 所有时刻的隐藏状态
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # 输入投影
        x = self.input_proj(x)
        
        # === sLSTM 初始化 ===
        h_s = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        c_s = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        n_s = torch.ones(batch, self.hidden_dim, device=device, dtype=dtype)
        
        # === mLSTM 初始化 ===
        C_m = torch.zeros(batch, self.hidden_dim, self.hidden_dim, device=device, dtype=dtype)
        n_m = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        
        all_h = []
        
        for t in range(seq_len):
            x_t = x[:, t]
            
            # Layer 1: sLSTM + 残差
            h_s, c_s, n_s = self.slstm_cell(x_t, h_s, c_s, n_s)
            h_s_out = self.slstm_norm(h_s)
            h_s_out = self.slstm_dropout(h_s_out)
            
            # Layer 2: mLSTM
            h_m, C_m, n_m = self.mlstm_cell(h_s_out, C_m, n_m)
            h_m_out = self.mlstm_dropout(h_m)
            
            # 门控融合: 结合 sLSTM 和 mLSTM 的输出
            gate = self.fusion_gate(torch.cat([h_s_out, h_m_out], dim=-1))
            h_fused = gate * h_m_out + (1 - gate) * h_s_out
            
            if return_all_states:
                all_h.append(h_fused)
        
        if return_all_states:
            return torch.stack(all_h, dim=1)
        else:
            return h_fused


class xLSTMEncoderV3(nn.Module):
    """xLSTM 混合编码器 V3
    
    官方风格：Pre-Norm + 残差连接的 sLSTM → mLSTM 串行堆叠。
    
    架构 (每个时间步):
        h = h + sLSTM(norm_s(h))   # sLSTM block with residual
        h = h + mLSTM(norm_m(h))   # mLSTM block with residual
    
    Args:
        input_dim: 输入维度 (嵌入维度)
        hidden_dim: 隐藏状态维度 (默认 64)
        dropout: Dropout 概率 (默认 0.1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        # sLSTM block: Pre-Norm + Cell + Dropout
        self.slstm_norm = nn.LayerNorm(hidden_dim)
        self.slstm_cell = sLSTMCell(hidden_dim, hidden_dim)
        self.slstm_dropout = nn.Dropout(dropout)
        
        # mLSTM block: Pre-Norm + Cell + Dropout
        self.mlstm_norm = nn.LayerNorm(hidden_dim)
        self.mlstm_cell = mLSTMCell(hidden_dim, hidden_dim)
        self.mlstm_dropout = nn.Dropout(dropout)
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_states: bool = False
    ) -> torch.Tensor:
        """编码输入序列
        
        Args:
            x: [batch, seq_len, input_dim] 输入序列
            return_all_states: 是否返回所有时刻的隐藏状态
        
        Returns:
            如果 return_all_states=False:
                [batch, hidden_dim] 最终隐藏状态
            如果 return_all_states=True:
                [batch, seq_len, hidden_dim] 所有时刻的隐藏状态
        """
        batch, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # 输入投影
        x = self.input_proj(x)
        
        # sLSTM 状态
        h_s = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        c_s = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        n_s = torch.ones(batch, self.hidden_dim, device=device, dtype=dtype)
        
        # mLSTM 状态
        C_m = torch.zeros(batch, self.hidden_dim, self.hidden_dim, device=device, dtype=dtype)
        n_m = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        
        all_h = []
        
        for t in range(seq_len):
            x_t = x[:, t]
            
            # === sLSTM Block: Pre-Norm + Residual ===
            h_normed = self.slstm_norm(x_t)
            h_s, c_s, n_s = self.slstm_cell(h_normed, h_s, c_s, n_s)
            h_s_out = self.slstm_dropout(h_s)
            h_res1 = x_t + h_s_out  # 残差连接（不影响 h_s 状态传递）
            
            # === mLSTM Block: Pre-Norm + Residual ===
            h_normed = self.mlstm_norm(h_res1)
            h_m, C_m, n_m = self.mlstm_cell(h_normed, C_m, n_m)
            h_m_out = self.mlstm_dropout(h_m)
            h_out = h_res1 + h_m_out  # 残差连接
            
            if return_all_states:
                all_h.append(h_out)
        
        # 输出归一化
        h_out = self.output_norm(h_out)
        
        if return_all_states:
            return torch.stack([self.output_norm(h_t) for h_t in all_h], dim=1)
        else:
            return h_out
