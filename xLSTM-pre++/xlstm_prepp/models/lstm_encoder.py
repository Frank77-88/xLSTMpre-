"""LSTM 编码器

标准 LSTM 编码器，用于与 sLSTM 对比实验。
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """LSTM 序列编码器
    
    使用标准 LSTM 编码输入序列，返回最终隐藏状态。
    接口与 sLSTMEncoder 保持一致，方便切换。
    
    Args:
        input_dim: 输入维度 (嵌入维度)
        hidden_dim: 隐藏状态维度 (默认 64)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
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
        # LSTM 前向传播
        outputs, (h_n, c_n) = self.lstm(x)
        
        if return_all_states:
            return outputs  # [batch, seq_len, hidden_dim]
        else:
            return h_n.squeeze(0)  # [batch, hidden_dim]
