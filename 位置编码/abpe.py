import torch
import math

def get_sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    # 推荐写法：直接使用幂运算 (pow)
    div_term = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# 用法：x = x + get_sinusoidal_pe(x.size(1), x.size(2))