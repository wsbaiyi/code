
import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super().__init__()
        assert d_model%num_head==0

        self.d_model=d_model
        self.num_head=num_head
        self.head_dim=d_model//num_head

        self.qkv=nn.Linear(self.d_model,3*self.d_model)
        self.out=nn.Linear(self.d_model,self.d_model)
    def forward(self,x,mask=None):
        B,T,C=x.shape
        # B,T,3*C
        inputs=self.qkv(x)
        q,k,v=inputs.chunk(3,dim=-1)

        q=q.view(B,T,self.num_head,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.num_head,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.num_head,self.head_dim).transpose(1,2)

        score=(q@k.transpose(-1,-2))/math.sqrt(self.head_dim)

        # mask
        mask=torch.tril(torch.ones(T,T,device=x.device),diagonal=0)

        score=score.masked_fill(mask==0,float('-inf'))

        score=f.softmax(score,dim=-1)

        output=(score@v).transpose(2,1).contiguous().view(B,T,C)

        return self.out(output)



    

class GroupQueryAttention(nn.Module):
    def __init__(self,d_model,q_head,kv_head):
        super().__init__()

        self.group=q_head//kv_head

        self.d_model=d_model
        self.q_head=q_head
        self.kv_head=kv_head
        self.head_dim=d_model//q_head

        self.q=nn.Linear(self.d_model,self.q_head*self.head_dim)
        self.k=nn.Linear(self.d_model,self.kv_head*self.head_dim)
        self.v=nn.Linear(self.d_model,self.kv_head*self.head_dim)

        self.out=nn.Linear(self.d_model,self.d_model)
    def forward(self,x,mask=None):
        B,T,C=x.shape
        # B,T,3*C
        
        q=self.q(x).view(B,T,self.q_head,self.head_dim).transpose(1,2)
        k=self.k(x).view(B,T,self.kv_head,self.head_dim).transpose(1,2)
        v=self.v(x).view(B,T,self.kv_head,self.head_dim).transpose(1,2)

        # 对齐qkv维度
        k=k.repeat_interleave(self.group,dim=1)
        v=v.repeat_interleave(self.group,dim=1)


        score=(q@k.transpose(-1,-2))/math.sqrt(self.head_dim)

        # mask
        mask=torch.tril(torch.ones(T,T,device=x.device),diagonal=0)

        score=score.masked_fill(mask==0,float('-inf'))

        score=f.softmax(score,dim=-1)

        output=(score@v).transpose(2,1).contiguous().view(B,T,C)

        return self.out(output)

