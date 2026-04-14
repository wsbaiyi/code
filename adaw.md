```python 
import torch

# 1. 初始化一个玩具参数 (比如大模型里的一个权重)
w = torch.tensor([10.0], requires_grad=True)

# 2. AdamW 的核心超参数
lr = 0.1             # 学习率
weight_decay = 0.01  # 权重衰减系数 (L2 正则化)
beta1, beta2 = 0.9, 0.999
eps = 1e-8

# 初始化动量状态
m = torch.zeros_like(w)
v = torch.zeros_like(w)
t = 0 # 记录步数

print(f"初始权重: {w.item():.4f}")

# 3. 模拟训练 5 步
for step in range(1, 6):
    t += 1
    
    # 模拟前向传播和计算 Loss (这里假设 Loss = w^2 / 2)
    # 注意：我们不在 Loss 里加 L2 正则化项！
    loss = 0.5 * w ** 2 
    
    # 计算纯净的梯度 (g_t)
    loss.backward()
    grad = w.grad.data
    
    with torch.no_grad(): # 更新参数时不追踪梯度
        # ==========================================
        # 🔥 核心亮点：AdamW 的解耦权重衰减 (Decoupled Weight Decay)
        # 直接在原来的权重上减去衰减值，完全不经过动量分母的稀释！
        # ==========================================
        w.data = w.data - lr * weight_decay * w.data
        
        # 下面是原汁原味的 Adam 动量计算
        # 1. 一阶动量 (方向)
        m = beta1 * m + (1 - beta1) * grad
        # 2. 二阶动量 (步长)
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # 3. 偏差修正 (Bias Correction)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # 4. 最终的梯度更新
        w.data = w.data - lr * m_hat / (torch.sqrt(v_hat) + eps)
    
    # 清空梯度
    w.grad.zero_()


import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义一个极简模型 (比如一个线性层)
model = nn.Linear(10, 2)

# 2. 声明 AdamW 优化器
# 注意：在大模型微调中，weight_decay 通常设置在 0.01 到 0.1 之间
optimizer = optim.AdamW(
    model.parameters(), 
    lr=1e-4, 
    weight_decay=0.01, # 直接在这里指定解耦的衰减率
    betas=(0.9, 0.999)
)

criterion = nn.MSELoss()
dummy_input = torch.randn(4, 10)
dummy_target = torch.randn(4, 2)

# 3. 标准的训练 Loop
for epoch in range(3):
    optimizer.zero_grad()       # 清空上一步梯度
    
    output = model(dummy_input) # 前向传播
    loss = criterion(output, dummy_target) # 计算 Loss (不需要自己手动写 L2 Penalty!)
    
    loss.backward()             # 反向传播算梯度
    
    # 这一步内部，PyTorch 会自动为你执行上面我手写的那套“解耦衰减 + 动量更新”逻辑
    optimizer.step()            
    
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")






```