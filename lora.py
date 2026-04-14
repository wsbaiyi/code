import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16):
        super().__init__()
        
        # 1. 原始的预训练权重 (全连接层)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # 🔥 核心亮点：冻结原始权重，严禁其参与梯度更新！
        self.linear.weight.requires_grad = False 
        
        # 2. LoRA 的两个低秩矩阵
        # 降维矩阵 A: [r, in_features]
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        # 升维矩阵 B: [out_features, r]
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # 3. LoRA 的缩放因子 (Scaling)
        # 类似于学习率的乘数，保证当我们改变秩 r 时，梯度的量级保持稳定
        self.scaling = lora_alpha / r
        
        # 4. 执行初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 🔥 核心亮点：A 用 Kaiming 正态分布初始化，B 用全零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始链路的输出
        original_out = self.linear(x)
        
        # LoRA 旁路的输出 (x 乘 A，再乘 B)
        # 公式: x * A^T * B^T * scaling
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # 两路结果相加
        return original_out + lora_out

# 测试一下我们手写的 LoRA
if __name__ == "__main__":
    # 模拟输入：Batch=2, 特征维度=1024
    x = torch.randn(2, 1024)
    
    # 实例化 LoRA 层 (输出维度=4096，秩 r=8)
    layer = LoRALinear(in_features=1024, out_features=4096, r=8)
    
    # 前向传播
    output = layer(x)
    
    print(f"输出形状: {output.shape}") 
    # 应该输出 [2, 4096]，且刚初始化时 output 严格等于 self.linear(x)