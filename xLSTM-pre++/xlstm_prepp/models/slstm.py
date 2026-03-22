"""sLSTM 编码器

实现论文 Eq.1-2: sLSTM with exponential gating and normalizer state

关键特性:
1. 输入门使用指数激活: i_t = exp(...)
2. 维护归一化状态 n_t 用于稳定隐藏状态计算
3. 隐藏输出: h_t = o_t * (c_t / n_t)
"""

import torch
import torch.nn as nn
from typing import Tuple


class sLSTMCell(nn.Module):
    """sLSTM 单元
    
    使用指数门控和归一化状态的 LSTM 变体。
    
    门控公式:
        z̃_t = W_z·x_t + R_z·h_{t-1} + b_z
        ĩ_t = W_i·x_t + R_i·h_{t-1} + b_i
        f̃_t = W_f·x_t + R_f·h_{t-1} + b_f
        õ_t = W_o·x_t + R_o·h_{t-1} + b_o
        
        z_t = tanh(z̃_t)
        i_t = exp(ĩ_t)      # 指数激活 (关键!)
        f_t = sigmoid(f̃_t)
        o_t = sigmoid(õ_t)
    
    状态更新:
        c_t = f_t * c_{t-1} + i_t * z_t
        n_t = f_t * n_{t-1} + i_t  # 归一化状态
        h̃_t = c_t / n_t
        h_t = o_t * h̃_t
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏状态维度
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入权重 (无偏置，偏置单独定义)
        self.W_z = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_i = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_f = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # 循环权重
        self.R_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.R_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.R_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.R_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 偏置
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        self.b_f = nn.Parameter(torch.ones(hidden_dim))  # 初始化为 1，利于梯度流
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self, 
        x_t: torch.Tensor, 
        h_prev: torch.Tensor, 
        c_prev: torch.Tensor, 
        n_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步前向传播
        
        Args:
            x_t: [batch, input_dim] 当前时刻输入
            h_prev: [batch, hidden_dim] 上一时刻隐藏状态
            c_prev: [batch, hidden_dim] 上一时刻细胞状态
            n_prev: [batch, hidden_dim] 上一时刻归一化状态
        
        Returns:
            h_t: [batch, hidden_dim] 当前隐藏状态
            c_t: [batch, hidden_dim] 当前细胞状态
            n_t: [batch, hidden_dim] 当前归一化状态
        """
        # 门控预激活
        z_tilde = self.W_z(x_t) + self.R_z(h_prev) + self.b_z
        i_tilde = self.W_i(x_t) + self.R_i(h_prev) + self.b_i
        f_tilde = self.W_f(x_t) + self.R_f(h_prev) + self.b_f
        o_tilde = self.W_o(x_t) + self.R_o(h_prev) + self.b_o
        
        # 激活函数
        z_t = torch.tanh(z_tilde)           # 细胞输入
        i_t = torch.exp(i_tilde)            # 指数输入门 (关键!)
        f_t = torch.sigmoid(f_tilde)        # 遗忘门
        o_t = torch.sigmoid(o_tilde)        # 输出门
        
        # 为了数值稳定性，限制 i_t 的最大值
        i_t = torch.clamp(i_t, max=50.0)
        
        # 状态更新
        c_t = f_t * c_prev + i_t * z_t      # 细胞状态
        n_t = f_t * n_prev + i_t            # 归一化状态
        
        # 归一化输出
        h_tilde = c_t / (n_t + 1e-6)        # 防止除零
        h_t = o_t * h_tilde
        
        return h_t, c_t, n_t


class sLSTMEncoder(nn.Module):
    """sLSTM 序列编码器
    
    使用 sLSTMCell 编码输入序列，返回最终隐藏状态。
    
    Args:
        input_dim: 输入维度 (嵌入维度)
        hidden_dim: 隐藏状态维度 (默认 64)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = sLSTMCell(input_dim, hidden_dim)
    
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
        
        # 初始化状态
        h = torch.zeros(batch, self.hidden_dim, device=device)
        c = torch.zeros(batch, self.hidden_dim, device=device)
        n = torch.ones(batch, self.hidden_dim, device=device)
        
        all_h = []
        
        # 逐步编码
        for t in range(seq_len):
            h, c, n = self.cell(x[:, t], h, c, n)
            if return_all_states:
                all_h.append(h)
        
        if return_all_states:
            return torch.stack(all_h, dim=1)  # [batch, seq_len, hidden_dim]
        else:
            return h  # [batch, hidden_dim]
    
    def get_input_gate_values(self, x: torch.Tensor) -> torch.Tensor:
        """获取输入门的值 (用于测试)
        
        Args:
            x: [batch, seq_len, input_dim] 输入序列
        
        Returns:
            [batch, seq_len, hidden_dim] 所有时刻的输入门值
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        h = torch.zeros(batch, self.hidden_dim, device=device)
        c = torch.zeros(batch, self.hidden_dim, device=device)
        n = torch.ones(batch, self.hidden_dim, device=device)
        
        all_i = []
        
        for t in range(seq_len):
            x_t = x[:, t]
            
            # 计算输入门
            i_tilde = self.cell.W_i(x_t) + self.cell.R_i(h) + self.cell.b_i
            i_t = torch.exp(i_tilde)
            i_t = torch.clamp(i_t, max=50.0)
            all_i.append(i_t)
            
            # 更新状态
            h, c, n = self.cell(x_t, h, c, n)
        
        return torch.stack(all_i, dim=1)
    
    def get_normalizer_states(self, x: torch.Tensor) -> torch.Tensor:
        """获取归一化状态 (用于测试)
        
        Args:
            x: [batch, seq_len, input_dim] 输入序列
        
        Returns:
            [batch, seq_len, hidden_dim] 所有时刻的归一化状态
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        h = torch.zeros(batch, self.hidden_dim, device=device)
        c = torch.zeros(batch, self.hidden_dim, device=device)
        n = torch.ones(batch, self.hidden_dim, device=device)
        
        all_n = []
        
        for t in range(seq_len):
            h, c, n = self.cell(x[:, t], h, c, n)
            all_n.append(n)
        
        return torch.stack(all_n, dim=1)
