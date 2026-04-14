```python 

import torch

def compute_grpo_loss(
    log_probs: torch.Tensor,      # 当前策略模型的 log 概率: [G, T]
    old_log_probs: torch.Tensor,  # 采样时旧模型的 log 概率: [G, T]
    ref_log_probs: torch.Tensor,  # 参考模型的 log 概率: [G, T]
    rewards: torch.Tensor,        # G 个回答的标量奖励: [G]
    epsilon: float = 0.2,         # PPO 截断阈值
    beta: float = 0.01            # KL 惩罚系数
):
    """
    输入参数说明:
    G: Group size (每个 Prompt 采样的回答组大小)
    T: Sequence length (每个回答的 token 数量)
    """
    
    # ==========================================
    # 步骤 1: 估算组相对优势 (Group Relative Advantage)
    # ==========================================
    # 公式: \hat{A}_i = (r_i - \mu_r) / (\sigma_r + \epsilon)
    mean_reward = rewards.mean()
    # 使用有偏标准差 (unbiased=False)，因为我们是在标准化当前这个确定的组
    std_reward = rewards.std(unbiased=False) 
    
    # 计算组内 Advantage [G]
    advantages = (rewards - mean_reward) / (std_reward + 1e-8) 
    
    # 广播到 token 级别，使其形状变为 [G, T]
    # 因为同一个回答里的所有 token 共享这一个句子级别的 Advantage
    advantages = advantages.unsqueeze(1).expand_as(log_probs)

    # ==========================================
    # 步骤 2: 计算策略比率 (Policy Ratio)
    # ==========================================
    # 公式: \rho_{i,t} = \pi_\theta / \pi_{\theta_{old}} 
    # 在对数空间中: exp(log_a - log_b) = a / b
    ratio = torch.exp(log_probs - old_log_probs) # [G, T]

    # ==========================================
    # 步骤 3: 计算 PPO 截断损失 (Clipping Loss)
    # ==========================================
    # 公式: \min( \rho \hat{A}, clip(\rho, 1-\epsilon, 1+\epsilon) \hat{A} )
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    clip_loss = torch.min(surr1, surr2) # [G, T]

    # ==========================================
    # 步骤 4: 计算严格为正且无偏的 KL 惩罚 (Unbiased KL Penalty)
    # ==========================================
    # 公式: \pi_{ref}/\pi_\theta - \log(\pi_{ref}/\pi_\theta) - 1
    # 设 \log x = \log \pi_{ref} - \log \pi_\theta
    log_x = ref_log_probs - log_probs
    x = torch.exp(log_x) # 这就是 \pi_{ref} / \pi_\theta
    
    kl_penalty = x - log_x - 1.0 # [G, T]

    # ==========================================
    # 步骤 5: 计算最终的总 Loss
    # ==========================================
    # 公式: \sum ( L_{clip} - \beta \mathbb{D}_{KL} )
    token_loss = clip_loss - beta * kl_penalty # [G, T]

    # 强化学习的目标是“最大化” token_loss
    # 在 PyTorch 中使用梯度下降，所以我们需要“最小化负的 token_loss”
    loss = -token_loss.mean()

    return loss






beta=0.1
clip=0.95

def grpo(log_prob,old_log_prob,ref_log_prob,reward,g):
    # 计算group advantage
    mean=reward.mean()
    std=reward.var(unbiased=False)

    advangtage=(reward-mean)/(std+1e-5)
    advantage=advantage.unsqueeze(1).expand_as(log_prob)

    ratio=log_prob-old_log_prob

    sur1=ratio*advantage
    sur2=torch.clamp(ratio,1-clip,1+clip)*advantage

    x=torch.exp(ref_log_prob-log_prob)
    kl=x-torch.log(x)-1.0

    loss=-torch.min(sur1+sur2)+beta*kl

    return loss.mean()








```