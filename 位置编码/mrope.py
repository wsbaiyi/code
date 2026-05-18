import torch


# =========================================================
# 1. rotate_half
# =========================================================

def rotate_half(x):
    """
    输入:
        x: [..., head_dim]

    假设:
        x = [x0, x1, x2, x3]

    输出:
        [-x1, x0, -x3, x2]

    作用:
        构造二维旋转里的:
            [-y, x]
    """

    # 偶数维
    x_even = x[..., ::2]

    # 奇数维
    x_odd = x[..., 1::2]

    # stack:
    #
    # [x0, x2]
    # [x1, x3]
    #
    # ->
    #
    # [
    #   [-x1, x0],
    #   [-x3, x2]
    # ]
    #

    x = torch.stack(
        (-x_odd, x_even),
        dim=-1
    )

    # flatten:
    #
    # [
    #   [-x1, x0],
    #   [-x3, x2]
    # ]
    #
    # ->
    #
    # [-x1, x0, -x3, x2]
    #

    return x.flatten(-2)


# =========================================================
# 2. 构造 mRoPE 的 cos/sin
# =========================================================

def build_mrope_cache(
    temporal_ids,   # [seq_len]
    height_ids,     # [seq_len]
    width_ids,      # [seq_len]
    head_dim
):
    """
    输入:

        temporal_ids:
            每个 token 属于第几帧

        height_ids:
            每个 token 在图像里的行坐标

        width_ids:
            每个 token 在图像里的列坐标

    例如:

        视频 patch:

        frame0:
            (0,0) (0,1)
            (1,0) (1,1)

        frame1:
            ...

    """

    device = temporal_ids.device

    # =====================================================
    # 2.1 每个轴占多少维
    # =====================================================

    #
    # 假设:
    #
    # head_dim = 12
    #
    # temporal -> 4维
    # height   -> 4维
    # width    -> 4维
    #

    section = head_dim // 3

    #
    # 由于 RoPE 是两维一组:
    #
    # (x0,x1)
    # (x2,x3)
    #
    # 所以真正频率数量:
    #

    half_section = section // 2


    # =====================================================
    # 2.2 构造频率
    # =====================================================

    #
    # shape:
    #
    # [half_section]
    #
    # 例如:
    #
    # [1.0, 0.01]
    #

    # / half_section 是为了让频率数量和 head_dim 匹配，本来分子也要乘以2，但可以省略
    inv_freq = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                half_section,
                device=device
            ).float()
            / half_section
        )
    )


    # =====================================================
    # 2.3 temporal 的 theta
    # =====================================================

    #
    # temporal_ids:
    # [seq_len]
    #
    # inv_freq:
    # [half_section]
    #
    # outer:
    #
    # ->
    #
    # [seq_len, half_section]
    #

    t_theta = torch.outer(
        temporal_ids.float(),
        inv_freq
    )

    h_theta = torch.outer(
        height_ids.float(),
        inv_freq
    )

    w_theta = torch.outer(
        width_ids.float(),
        inv_freq
    )


    # =====================================================
    # 2.4 cos/sin
    # =====================================================

    #
    # shape:
    #
    # [seq_len, half_section]
    #

    t_cos = torch.cos(t_theta)
    t_sin = torch.sin(t_theta)

    h_cos = torch.cos(h_theta)
    h_sin = torch.sin(h_theta)

    w_cos = torch.cos(w_theta)
    w_sin = torch.sin(w_theta)


    # =====================================================
    # 2.5 repeat_interleave
    # =====================================================

    #
    # 因为:
    #
    # (x0,x1)
    #
    # 共用一个角度
    #
    # 所以:
    #
    # [a,b]
    #
    # ->
    #
    # [a,a,b,b]
    #

    t_cos = torch.repeat_interleave(t_cos, 2, dim=-1)
    t_sin = torch.repeat_interleave(t_sin, 2, dim=-1)

    h_cos = torch.repeat_interleave(h_cos, 2, dim=-1)
    h_sin = torch.repeat_interleave(h_sin, 2, dim=-1)

    w_cos = torch.repeat_interleave(w_cos, 2, dim=-1)
    w_sin = torch.repeat_interleave(w_sin, 2, dim=-1)


    # =====================================================
    # 2.6 拼接
    # =====================================================

    #
    # 最终:
    #
    # cos:
    #
    # [
    #   temporal部分,
    #   height部分,
    #   width部分
    # ]
    #
    # shape:
    #
    # [seq_len, head_dim]
    #

    cos = torch.cat(
        [t_cos, h_cos, w_cos],
        dim=-1
    )

    sin = torch.cat(
        [t_sin, h_sin, w_sin],
        dim=-1
    )

    return cos, sin


# =========================================================
# 3. apply mRoPE
# =========================================================

def apply_mrope(
    q,      # [B, S, H, D]
    k,      # [B, S, H, D]
    cos,    # [S, D]
    sin     # [S, D]
):

    #
    # unsqueeze:
    #
    # [S,D]
    #
    # ->
    #
    # [1,S,1,D]
    #
    # 为了 broadcast
    #

    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    #
    # 真正 RoPE:
    #

    q_out = (
        q * cos
        + rotate_half(q) * sin
    )

    k_out = (
        k * cos
        + rotate_half(k) * sin
    )

    return q_out, k_out