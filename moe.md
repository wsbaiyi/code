```python

import torch
import torch.nn as nn
import torch.nn.functional as F

class InterviewMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k, use_shared=True):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        
        # 1. 门控 (Router)
        self.gate = nn.Linear(dim, num_experts)
        
        # 2. 专家群 (Experts)
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts) # 用简单层代替FFN示意
        ])
        
        # 3. 亮点：加上 Shared Expert (只需一行，显示你懂 DeepSeek/MLA)
        self.use_shared = use_shared
        if use_shared:
            self.shared_expert = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        # --- Step 1: 路由逻辑 ---
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        
        # 亮点：口述这里需要计算 Aux Loss (Load Balancing)
        # self.compute_aux_loss(probs) 
        
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # 归一化权重
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # --- Step 2: 专家计算 (面试版核心) ---
        # 声明：这里为了白板可读性，我使用 Mask 方式。
        # 实际生产中这里会用 argsort + group gemm 优化显存和速度。
        
        final_output = torch.zeros_like(x_flat)
        
        # 遍历所有专家 (逻辑最清晰，不容易写挂)
        for i in range(self.num_experts):
            # 这是一个 mask，找出当前 batch 中所有分配给专家 i 的 token
            # 只要 topk 的索引里包含了 i，就要计算
            # indices shape: (N, k)
            # idx_mask  shape :(N,k)
            # mask.shape:  (B * S,)
            idx_mask = (topk_indices == i) 
            mask = idx_mask.any(dim=-1) 
            
            if mask.any():
                # 1. 选出数据 (正确)
                # mask shape: (B*S,), 选出 M 个需要计算的 token
                selected_x = x_flat[mask] 
                
                # 2. 专家计算 (正确)
                # expert_out shape: (M, D)
                expert_out = self.experts[i](selected_x)
                
                # 3. 取出权重 (正确)
                # idx_mask shape: (B*S, K), topk_weights shape: (B*S, K)
                # 这一步利用布尔索引取出对应的 M 个权重值 -> shape: (M,)
                weights_for_this_expert = topk_weights[idx_mask]
                
                # 4. 乘权重并累加 (!!! 核心修改点 !!!)
                
                # 修改点 A: 增加维度 .view(-1, 1) 以便广播
                # (M, D) * (M, 1) -> (M, D)
                weighted_expert_out = expert_out * weights_for_this_expert.view(-1, 1)
                
                # 修改点 B: 使用 mask (行索引) 而不是 idx_mask
                # final_output[mask] 选出的是那 M 行，形状 (M, D)
                final_output[mask] += weighted_expert_out
        # --- Step 3: Shared Expert (加分项) ---
        if self.use_shared:
            final_output += self.shared_expert(x_flat)

        return final_output.view(B, S, D)

















class Moe(nn.Module):
    def __init__(self,topk,num_expert,dim,shared=True):
        super().__init__()
        self.topk=topk
        self.shared=shared

        self.gate=nn.Linear(dim,num_expert)
        self.experts=nn.ModuleList(
            [nn.Linear(dim,dim) for _ in range(num_expert)]
        )
        self.num_expert=num_expert
        if self.shared:
            self.shared_expert=nn.Linear(dim,dim)
    def forward(self,x):
        B,S,D=x.shape

        x_flat=x.reshape(-1,D)
        # B*S,expert
        logits=self.gate(x_flat)
        logits=logits.softmax(dim=-1)

        # B*S,topk
        weight,idx=torch.topk(logits,self.topk)

        weight=weight/weight.sum(dim=-1,keepdim=True)


        out=torch.zeros_like(x_flat)

        for i in range(self.num_expert):
            idx_mask=(idx==i)
            mask=idx_mask.any(dim=-1)

            if mask.any():

                # 得到x   M,dim
                selected_x=x_flat[mask]
                # M,dim
                output=self.experts[i](selected_x)

                # weight计算;  M,1
                selected_weight=weight[idx_mask].view(-1,1)
                # M,d
                out[mask]+=selected_weight*output
        if self.shared:
            out+=self.shared_expert(x_flat)
        
        return out.reshape(B,S,D)



def compute_aux_loss(logits, top_k_indices, num_experts):
    """
    logits: (Batch * Seq, Num_Experts)
    top_k_indices: (Batch * Seq, TopK), 其中 TopK > 1
    num_experts: int
    """
    # top_k_indices 的形状是 (B*S, k)
    # k 就是 top_k 的大小
    k = top_k_indices.shape[-1]

    # --- 1. 计算 P (Probability) ---
    # Router 想要分配给专家的概率分布
    # (B*S, N)
    # 每行总和为1，总和为N
    probs = F.softmax(logits, dim=-1)
    
    # 在 Batch 维度求均值，得到 Router 对每个专家的平均“喜爱程度”
    # P shape: (N,)， sum(P) = 1
    # mean(dim=0)作用是每行求和除以行数，也就是/N
    # 总和为1
    P = probs.mean(dim=0)

    # --- 2. 计算 f (Frequency) ---
    # 实际专家被选中的频率。
    # 我们需要计算：在所有的路由决策（共 B*S*k 次）中，专家 i 被选中了多少次？
    
    # 我们先创建一个全 0 的 mask: (B*S, N)
    # 这里的逻辑是将 k 个选择压缩到 (B*S, N) 的每一行中
    
    mask = torch.zeros_like(probs)
    
    # scatter_(dim, index, src)
    # dim=1: 沿着专家维度填值
    # index=top_k_indices: 告诉它哪些位置要填 1
    # value=1.0: 填入的值
    # 每行总和为k，总和为N*k
    mask.scatter_(1, top_k_indices, 1.0) # (B*S, N)
    
    # 现在 mask 的每一行有 k 个 1 (表示选中了 k 个专家)
    # 求均值得到 f
    # f shape: (N,)。注意：此时 sum(f) = k，因为每个样本投了 k 票
    # mean(dim=0)作用是每行求和除以行数，也就是N*k/N
    f = mask.mean(dim=0)
    
    # --- 3. 归一化与 Loss 计算 ---
    # P 的和是 1， f 的和是 k。
    # 两个向量做点积，如果不归一化，Loss 会随着 k 的增大而线性增大。
    # 为了让 Loss 对 k 不敏感，通常将 f 除以 k，使其和也为 1。
    f = f / k 
    
    # 计算点积 Loss
    # sum(P * f) * N * alpha
    aux_loss = (P * f).sum() * num_experts
    
    return aux_loss


def compute_sequence_level_aux_loss(logits, top_k_indices, num_experts):
    """
    logits: (Batch, Seq, N)  <- 注意这里保持 3 维
    top_k_indices: (Batch, Seq, k)
    """
    k = top_k_indices.shape[-1]
    
    # 1. 计算 P (Probability)
    probs = F.softmax(logits, dim=-1) # (B, S, N)
    
    # 关键点：只在 dim=1 (Seq维度) 求均值，保留 Batch 维度
    # P_seq: (B, N) -> 每一条数据内部，Router 想分配的分布
    P_seq = probs.mean(dim=1) 

    # 2. 计算 f (Frequency)
    # 使用 scatter 在 sequence 维度统计频率
    mask = torch.zeros_like(probs) # (B, S, N)
    mask.scatter_(2, top_k_indices, 1.0) # 在 dim=2 (专家维度) 填 1
    
    # 关键点：在 dim=1 (Seq维度) 求均值
    # f_seq: (B, N) -> 每一条数据内部，实际选中的分布
    f_seq = mask.mean(dim=1)
    
    # 归一化 (同上)
    f_seq = f_seq / k

    # 3. 计算 Loss
    # 现在我们有 (B, N) 的 P 和 f
    # 先算每条数据的点积 sum(dim=-1) -> (B,)
    loss_per_seq = (P_seq * f_seq).sum(dim=-1) * num_experts
    
    # 最后对 Batch 求平均
    return loss_per_seq.mean()











# logits: B*S,D
# idx: B*S，K
def batch_level_aux_loss(logits,idx,num_expert):
    k=idx.size(-1)

    # softmax
    logits=logits.softmax(dim=-1)

    # 1. 计算p；想要分配的; 总和为1
    # (D,)
    probs=logits.mean(dim=0)

    # 2. 计算f；实际分配的
    f=torch.zeros_like(logits)
    # B*S,D
    # 总和为N*k
    f.scatter_(dim=1,index=idx,value=1.0)   
    f=f.mean(dim=0)/k

    # 计算总和
    aux_loss=(f*p).sum()*num_expert
    return aux_loss


# logits: B,S,D
# idx: B, S，K
def sequence_level_aux_loss(logits,idx,num_expert):
    k=idx.size(-1)

    # softmax
    logits=logits.softmax(dim=-1)

    # 1. 计算p；想要分配的; 总和为1
    # (B,D)
    probs=logits.mean(dim=1)

    # 2. 计算f；实际分配的
    f=torch.zeros_like(logits)
    # B,S,D
    # 总和为N*k
    f.scatter_(dim=2,index=idx,value=1.0)   
    # B,D
    f=f.mean(dim=1)/k

    # 计算总和
    aux_loss=((f*p).sum(dim=-1)*num_expert).mean()
    return aux_loss




















import torch
import torch.nn as nn
import torch.nn.functional as F

class InterviewMoE(nn.Module):
    def __init__(self, dim, num_experts, top_k, use_shared=True):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        
        # 1. 门控 (Router)
        self.gate = nn.Linear(dim, num_experts)
        
        # 2. 专家群 (Experts)
        self.experts = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_experts) # 用简单层代替FFN示意
        ])
        
        # 3. 亮点：加上 Shared Expert (只需一行，显示你懂 DeepSeek/MLA)
        self.use_shared = use_shared
        if use_shared:
            self.shared_expert = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (batch, seq, dim)
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        
        # --- Step 1: 路由逻辑 ---
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        
        # 亮点：口述这里需要计算 Aux Loss (Load Balancing)
        aux_loss=self.compute_aux_loss(probs) 
        
        topk_weights, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        # 归一化权重
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # --- Step 2: 专家计算 (面试版核心) ---
        # 声明：这里为了白板可读性，我使用 Mask 方式。
        # 实际生产中这里会用 argsort + group gemm 优化显存和速度。
        
        final_output = torch.zeros_like(x_flat)
        
        # 遍历所有专家 (逻辑最清晰，不容易写挂)
        for i in range(self.num_experts):
            # 这是一个 mask，找出当前 batch 中所有分配给专家 i 的 token
            # 只要 topk 的索引里包含了 i，就要计算
            # indices shape: (N, k)
            # idx_mask  shape :(N,k)
            # mask.shape:  (B * S,)
            idx_mask = (topk_indices == i) 
            mask = idx_mask.any(dim=-1) 
            
            if mask.any():
                # 1. 选出数据 (正确)
                # mask shape: (B*S,), 选出 M 个需要计算的 token
                selected_x = x_flat[mask] 
                
                # 2. 专家计算 (正确)
                # expert_out shape: (M, D)
                expert_out = self.experts[i](selected_x)
                
                # 3. 取出权重 (正确)
                # idx_mask shape: (B*S, K), topk_weights shape: (B*S, K)
                # 这一步利用布尔索引取出对应的 M 个权重值 -> shape: (M,)
                weights_for_this_expert = topk_weights[idx_mask]
                
                # 4. 乘权重并累加 (!!! 核心修改点 !!!)
                
                # 修改点 A: 增加维度 .view(-1, 1) 以便广播
                # (M, D) * (M, 1) -> (M, D)
                weighted_expert_out = expert_out * weights_for_this_expert.view(-1, 1)
                
                # 修改点 B: 使用 mask (行索引) 而不是 idx_mask
                # final_output[mask] 选出的是那 M 行，形状 (M, D)
                final_output[mask] += weighted_expert_out
        # --- Step 3: Shared Expert (加分项) ---
        if self.use_shared:
            final_output += self.shared_expert(x_flat)

        return final_output.view(B, S, D),aux_loss




































```