"""mLSTM 编码器

实现矩阵记忆的 LSTM 变体，具有更大的记忆容量。

关键特性:
1. 矩阵记忆状态 C ∈ R^{d×d}，容量为 O(d²)
2. 使用 query-key-value 机制，类似线性注意力
3. 指数门控 + 归一化状态确保稳定性
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class mLSTMCell(nn.Module):
    """mLSTM 单元
    
    使用矩阵记忆的 LSTM 变体，记忆容量从 O(d) 提升到 O(d²)。
    
    核心公式:
        q_t = W_q·x_t    (query)
        k_t = W_k·x_t    (key)  
        v_t = W_v·x_t    (value)
        
        i_t = exp(w_i·x_t + b_i)     # 指数输入门
        f_t = sigmoid(w_f·x_t + b_f) # 遗忘门
        o_t = sigmoid(W_o·x_t + b_o) # 输出门
        
        C_t = f_t * C_{t-1} + i_t * (v_t ⊗ k_t)  # 矩阵记忆更新
        n_t = f_t * n_{t-1} + i_t * k_t           # 归一化状态
        
        h̃_t = C_t · q_t / max(|n_t · q_t|, 1)    # 归一化检索
        h_t = o_t * h̃_t
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏状态维度
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value 投影
        self.W_q = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # 门控 (输入门和遗忘门使用简化的向量门控)
        self.w_i = nn.Linear(input_dim, 1, bias=True)
        self.w_f = nn.Linear(input_dim, 1, bias=True)
        self.W_o = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # 输出层归一化
        self.out_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        
        # 遗忘门偏置初始化为正值，利于梯度流
        nn.init.constant_(self.w_f.bias, 3.0)
        nn.init.zeros_(self.w_i.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        C_prev: torch.Tensor,
        n_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步前向传播
        
        Args:
            x_t: [batch, input_dim] 当前输入
            C_prev: [batch, hidden_dim, hidden_dim] 上一时刻矩阵记忆
            n_prev: [batch, hidden_dim] 上一时刻归一化状态
        
        Returns:
            h_t: [batch, hidden_dim] 当前隐藏状态
            C_t: [batch, hidden_dim, hidden_dim] 当前矩阵记忆
            n_t: [batch, hidden_dim] 当前归一化状态
        """
        # Query, Key, Value
        q_t = self.W_q(x_t)  # [batch, hidden_dim]
        k_t = self.W_k(x_t)  # [batch, hidden_dim]
        v_t = self.W_v(x_t)  # [batch, hidden_dim]
        
        # 门控
        i_t = torch.exp(self.w_i(x_t))  # [batch, 1]
        i_t = torch.clamp(i_t, max=50.0)  # 数值稳定性
        f_t = torch.sigmoid(self.w_f(x_t))  # [batch, 1]
        o_t = torch.sigmoid(self.W_o(x_t))  # [batch, hidden_dim]
        
        # 矩阵记忆更新: C_t = f_t * C_{t-1} + i_t * (v_t ⊗ k_t)
        # v_t ⊗ k_t 是外积: [batch, hidden_dim, 1] @ [batch, 1, hidden_dim]
        vk_outer = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)  # [batch, hidden_dim, hidden_dim]
        C_t = f_t.unsqueeze(-1) * C_prev + i_t.unsqueeze(-1) * vk_outer
        
        # 归一化状态更新: n_t = f_t * n_{t-1} + i_t * k_t
        n_t = f_t * n_prev + i_t * k_t  # [batch, hidden_dim]
        
        # 检索: h̃_t = C_t · q_t
        h_tilde = torch.bmm(C_t, q_t.unsqueeze(-1)).squeeze(-1)  # [batch, hidden_dim]
        
        # 归一化: 除以 max(|n_t · q_t|, 1)
        nq = torch.sum(n_t * q_t, dim=-1, keepdim=True)  # [batch, 1]
        normalizer = torch.clamp(torch.abs(nq), min=1.0)
        h_tilde = h_tilde / normalizer
        
        # 输出
        h_t = o_t * h_tilde
        h_t = self.out_norm(h_t)
        
        return h_t, C_t, n_t


class mLSTMEncoder(nn.Module):
    """mLSTM 序列编码器
    
    使用 mLSTMCell 编码输入序列，返回最终隐藏状态。
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏状态维度 (默认 64)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = mLSTMCell(input_dim, hidden_dim)
    
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
        
        # 初始化状态
        C = torch.zeros(batch, self.hidden_dim, self.hidden_dim, device=device, dtype=dtype)
        n = torch.zeros(batch, self.hidden_dim, device=device, dtype=dtype)
        
        all_h = []
        
        # 逐步编码
        for t in range(seq_len):
            h, C, n = self.cell(x[:, t], C, n)
            if return_all_states:
                all_h.append(h)
        
        if return_all_states:
            return torch.stack(all_h, dim=1)
        else:
            return h
