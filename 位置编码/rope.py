import torch


# -----------------------------------
# 1. 生成 cos/sin
# -----------------------------------

def build_rope_cache(seq_len, head_dim):

    # [0,1,2,...]
    position = torch.arange(seq_len).float()

    # 频率
    # shape: [head_dim//2]
    freq = 1.0 / (
        10000 ** (
            torch.arange(0, head_dim, 2).float() / head_dim
        )
    )

    # 外积
    # shape: [seq_len, head_dim//2]
    theta = torch.outer(position, freq)

    # cos/sin
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    # 扩展成 head_dim
    # [a,b] -> [a,a,b,b]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)

    return cos, sin


# -----------------------------------
# 2. rotate_half
# -----------------------------------

def rotate_half(x):

    # 偶数维
    x1 = x[..., ::2]

    # 奇数维
    x2 = x[..., 1::2]

    # [-x2, x1]
    x = torch.stack((-x2, x1), dim=-1)

    return x.flatten(-2)


# -----------------------------------
# 3. 应用 RoPE
# -----------------------------------

def apply_rope(x, cos, sin):

    # x:
    # [B, S, H, D]

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    return x * cos + rotate_half(x) * sin


# -----------------------------------
# 测试
# -----------------------------------

B, S, H, D = 2, 4, 8, 4

x = torch.randn(B, S, H, D)

cos, sin = build_rope_cache(S, D)

out = apply_rope(x, cos, sin)

print(out.shape)