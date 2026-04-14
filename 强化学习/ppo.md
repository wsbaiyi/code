```python
import torch
import torch.nn.functional as F

prompts = torch.randint(0, 10000, (4, 10))
beta = 0.8
clip_eps = 0.2
ppo_epochs = 4  # 🌟 核心新增：复用这批数据的次数

def ppo_train_loop(prompts, actor, critic, ref, reward, actor_opt, critic_opt):
    # ==========================================================
    # 阶段一：Rollout (攒经验阶段，绝对不能有梯度)
    # 目标：收集 {状态, 动作, 原始概率, 原始价值, 奖励, 优势}
    # ==========================================================
    with torch.no_grad():
        # 1. 演员上台表演
        responses = actor.generate(prompts)
        
        # 2. 拍下“案发现场”的快照
        actor_logits = actor(prompts, responses)
        ref_logits = ref(prompts, responses)
        
        # 提取对数概率并切断梯度，因为它们在后续的 Epoch 中是常数！
        log_actor_old = F.log_softmax(actor_logits, dim=-1).detach() 
        log_ref = F.log_softmax(ref_logits, dim=-1)
        log_kl = log_actor_old - log_ref
        
        env_rewards = reward(prompts, responses)
        final_rewards = env_rewards - beta * log_kl
        
        # 提取 Critic 的原始预测并切断梯度
        values_old = critic(prompts, responses).detach() 
        
        # 3. 计算优势 (Advantage) - 这也是个固定下来的常数
        advantages = (final_rewards - values_old).detach()
        
        # 💡 工程Trick：对优势函数做归一化，能极大提升多轮 Epoch 的训练稳定性
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 此时，阶段一结束。我们拿着上面算出的所有“常数”，进入阶段二。

    # ==========================================================
    # 阶段二：PPO Epochs (复盘学习阶段，开启梯度)
    # 目标：在保证 ratio 不超标的前提下，尽量榨干这批数据的价值
    # ==========================================================
    for epoch in range(ppo_epochs):
        
        # 1. 重新审视案发现场（当前最新权重下的 Actor 和 Critic）
        new_logits = actor(prompts, responses)
        new_log_probs = F.log_softmax(new_logits, dim=-1)
        
        new_values = critic(prompts, responses)

        # 2. 计算 Actor Loss (用最新的概率除以第一阶段冻结的老概率)
        ratio = torch.exp(new_log_probs - log_actor_old) 
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # 3. 计算 Critic Loss (让最新的预测值逼近第一阶段算出的真实奖励)
        critic_loss = F.mse_loss(new_values, final_rewards)

        # 4. 反向传播更新网络参数
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        
        print(f"Epoch {epoch+1}/{ppo_epochs} | Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f}")

    # 4 个 Epoch 结束后，这批 Prompt 和 Response 的价值被榨干，丢弃。
    # 准备接收下一个 Batch 的 Prompts，重新开始阶段一。
    return "一个完整的 Rollout -> 多轮 Update 循环结束"

```