import torch
import math
class LearnablePE(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # 就是一个简单的查表
        self.pe = torch.nn.Embedding(max_len, d_model)
    def forward(self, x):
        # x: [B, S, D]
        # positions : [S]
        positions = torch.arange(0, x.size(1), device=x.device)
        # self.pe(positions) : [S, D]
        return x + self.pe(positions)