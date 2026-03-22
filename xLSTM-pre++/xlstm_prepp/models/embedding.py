"""输入嵌入层

实现论文 Eq.3: Ψ(x; W_emb) = LeakyReLU(FC(x))
"""

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """输入嵌入层
    
    将原始输入特征转换为更高维度的嵌入表示。
    
    Ψ(x; W_emb) = LeakyReLU(FC(x))
    
    Args:
        input_dim: 输入维度 (X-TRAJ: 4, X-TRACK: 2)
        embed_dim: 嵌入维度 (默认 32)
        negative_slope: LeakyReLU 负斜率 (默认 0.1)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int = 32,
        negative_slope: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.fc = nn.Linear(input_dim, embed_dim)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [batch, seq_len, input_dim] 输入特征
        
        Returns:
            [batch, seq_len, embed_dim] 嵌入特征
        """
        return self.activation(self.fc(x))
