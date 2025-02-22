# Original code: https://github.com/feizc/DiT-MoE/blob/main/models.py

from dataclasses import dataclass
from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torch import Tensor

from transformer.math import attention

#################################################################################
#                                MoE Layer.                                     #
#################################################################################
class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0) 
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=-1)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class EC_MoEGate(nn.Module):
    """
    Expert-Choice gating without auxiliary loss.
    Each expert will pick the top-C tokens, where
      C = (S * f_c) / E,
    computed in the forward pass.
    """
    def __init__(self, embed_dim, num_experts=16, f_c=2.0):
        super().__init__()
        self.num_experts = num_experts
        self.f_c = f_c  # capacity factor

        # gating weight: shape (E, d)
        self.weight = nn.Parameter(torch.empty(num_experts, embed_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        """
        hidden_states shape: (B, S, d)
        Returns:
            topk_idx:    LongTensor of shape (B, E, C)
            topk_weight: FloatTensor of shape (B, E, C)
        """
        B, S, d = hidden_states.shape

        # Compute logits for each token: shape (B, S, E)
        logits = F.linear(hidden_states, self.weight)  # no bias

        # Apply softmax over experts for each token: shape (B, S, E)
        scores = logits.softmax(dim=-1)

        # Transpose to shape (B, E, S) so that for each batch and each expert we have S scores
        scores = scores.transpose(1, 2)  # now (B, E, S)

        # Compute per-example capacity: each expert picks top C tokens per example
        capacity = int(S * self.f_c / self.num_experts)

        # Each expert in each batch element picks top-C tokens over the S tokens
        topk_weight, topk_idx = torch.topk(scores, k=capacity, dim=-1, sorted=False)
        # topk_weight and topk_idx: (B, E, C)

        return topk_idx, topk_weight


class EC_SparseMoeBlock(nn.Module):
    """
    Expert-Choice sparse MoE block with no auxiliary load-balancing loss.
    """
    def __init__(
        self,
        embed_dim,
        mlp_dim=4,
        num_experts=16,
        f_c=2.0,
        pretraining_tp=2,
        num_shared_experts=2
    ):
        super().__init__()
        self.num_experts = num_experts
        self.f_c = f_c

        # Create experts
        self.experts = nn.ModuleList([
            MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=mlp_dim,
                pretraining_tp=pretraining_tp
            )
            for _ in range(num_experts)
        ])

        # Our updated gate with no aux loss
        self.gate = EC_MoEGate(
            embed_dim=embed_dim,
            num_experts=num_experts,
            f_c=f_c
        )

        # Shared "expert"
        self.n_shared_experts = num_shared_experts
        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=intermediate_size,
                pretraining_tp=pretraining_tp
            )

    def forward(self, hidden_states):
        """
        hidden_states: (B, S, d)
        """
        B, S, d = hidden_states.shape
        identity = hidden_states

        # Gate returns indices of shape (B, E, C)
        topk_idx, topk_weight = self.gate(hidden_states)

        # Create an output buffer for the flattened tokens: (B*S, d)
        flat_out = torch.zeros(B * S, d, device=hidden_states.device, dtype=hidden_states.dtype)

        # Compute an offset for each batch element so that indices point to the correct positions
        batch_offset = (torch.arange(B, device=hidden_states.device) * S).view(B, 1, 1)
        # Adjust the token indices to be in the flattened space: shape (B, E, C)
        flat_topk_idx = topk_idx + batch_offset

        # Flatten hidden_states: (B*S, d)
        flat_x = hidden_states.view(B * S, d)

        # Process each expert (the gate still has shape (B, E, C))
        for expert_idx, expert_layer in enumerate(self.experts):
            # Extract indices and weights for expert_idx across the entire batch: shapes (B, C)
            expert_indices = flat_topk_idx[:, expert_idx, :]  # (B, C)
            expert_weights = topk_weight[:, expert_idx, :]      # (B, C)

            # Flatten these so that we process all selected tokens at once: (B*C,)
            expert_indices_flat = expert_indices.reshape(-1)
            expert_weights_flat = expert_weights.reshape(-1, 1)  # (B*C, 1)

            # Gather tokens from the flattened hidden states: (B*C, d)
            expert_tokens = flat_x[expert_indices_flat]
            
            # Process the tokens through the expert
            expert_out = expert_layer(expert_tokens)
            
            # Weight the outputs by the gating weights
            expert_out = expert_out * expert_weights_flat

            # Scatter-add back into the flat output buffer
            flat_out.index_add_(0, expert_indices_flat, expert_out)

        # Reshape the output back to (B, S, d)
        out = flat_out.view(B, S, d)

        # Optionally add the shared-expert MLP
        if self.n_shared_experts is not None:
            out = out + self.shared_experts(identity)

        return out

### Token choice
class TC_MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape    
        # print(bsz, seq_len, h)    
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class TC_SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, embed_dim, mlp_dim, num_experts=16, num_experts_per_tok=2, pretraining_tp=2, num_shared_experts=2):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList([MoeMLP(hidden_size = embed_dim, intermediate_size = mlp_dim, pretraining_tp=pretraining_tp) for i in range(num_experts)])
        self.gate = TC_MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.n_shared_experts = num_shared_experts
        
        if self.n_shared_experts is not None:
            intermediate_size =  embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size = embed_dim, intermediate_size = intermediate_size, pretraining_tp=pretraining_tp)
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states) 
        # print(topk_idx.tolist(), print(len(topk_idx.tolist()))) 
        # global selected_ids_list
        # selected_ids_list.append(topk_idx.tolist())

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
    

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) 
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok 
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]]) 
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale
