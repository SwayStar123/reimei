# Original code: https://github.com/feizc/DiT-MoE/blob/main/models.py

from dataclasses import dataclass
from einops import rearrange
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from transformer.utils import remove_masked_tokens
from transformer.embed import rope_1d, rope_2d

#################################################################################
#                                MoE Layer.                                     #
#################################################################################
class MoEGate(nn.Module):
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
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
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


class SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """
    def __init__(self, embed_dim, mlp_ratio=4, num_experts=16, num_experts_per_tok=2, pretraining_tp=2, num_shared_experts=2):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList([MoeMLP(hidden_size = embed_dim, intermediate_size = mlp_ratio * embed_dim, pretraining_tp=pretraining_tp) for i in range(num_experts)])
        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.n_shared_experts = num_shared_experts
        
        if self.n_shared_experts is not None:
            intermediate_size =  embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(hidden_size = embed_dim, intermediate_size = intermediate_size, pretraining_tp=pretraining_tp)

        self.initialize_weights()

    def initialize_weights(self):
        self.gate.initialize_weights()
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states) 

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts): 
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i]).float()
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

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: torch.Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.gelu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
    

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout=0.0) -> torch.Tensor:
    """
    Scaled dot-product attention with shape conventions:
      q, k, v: [B, H, L, D]
        B = batch size
        H = number of heads
        L = sequence length
        D = head dimension
    Returns:
      x: [B, L, H*D], sequence-first
    """
    # Use PyTorch 2.0 built-in scaled_dot_product_attention
    # which expects q,k,v: [B, H, L, D]
    # and returns [B, H, L, D]
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)  # shape [B, H, L, D]
    
    # Rearrange to [B, L, H*D]
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

#################################################################################
#                                 Core DiT Modules                              #
#################################################################################

class DoubleStreamBlock(nn.Module):
    """
    A DiT block with MoE for text & image, plus separate 1D & 2D RoPE application
    and joint attention. Similar to flux implementation, but with MoE instead of MLP.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_experts=8,
        num_experts_per_tok=2,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        # Norm + SelfAttention for image
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.img_qk_norm = QKNorm(self.head_dim)
        self.img_proj = nn.Linear(hidden_size, hidden_size)

        # Norm + SelfAttention for text
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.txt_qk_norm = QKNorm(self.head_dim)
        self.txt_proj = nn.Linear(hidden_size, hidden_size)

        # MoE blocks instead of standard MLP
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_moe = SparseMoeBlock(
            hidden_size, mlp_ratio, num_experts, num_experts_per_tok, pretraining_tp, num_shared_experts
        )
        self.txt_moe = SparseMoeBlock(
            hidden_size, mlp_ratio, num_experts, num_experts_per_tok, pretraining_tp, num_shared_experts
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            # 1) Linear layers -> Xavier uniform
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # 2) LayerNorm -> weight=1, bias=0
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Apply basic initialization recursively to all submodules
        self.apply(_basic_init)

        self.img_moe.initialize_weights()
        self.txt_moe.initialize_weights()

    def forward(
        self,
        img: torch.Tensor,          # [B, L_img, hidden_size]
        txt: torch.Tensor,          # [B, L_txt, hidden_size]
        vec: torch.Tensor,          # conditioning vector => Modulation
        mask: torch.Tensor,
        h: int,
        w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns updated (img, txt).
        We do:
          1) image attn with MoE
          2) text attn with MoE
          3) joint attention of (txt+img)
        """
        B, L_txt, _ = txt.shape

        # (seq_len, pos_emb_dim)
        txt_rope = rope_1d(self.head_dim, L_txt)
        # (batch_size, seq_len, pos_emb_dim)
        txt_rope = txt_rope.unsqueeze(0).repeat(B, 1, 1).to(img.device)

        # (height, width, embed_dim)
        img_rope = rope_2d(self.head_dim, h, w)
        # (batch_size, height*width, pos_emb_dim)
        img_rope = img_rope.unsqueeze(0).repeat(B, 1, 1).to(img.device)
        if mask is not None:
            img_rope = remove_masked_tokens(img_rope, mask)

        # 1) modulate image
        img_mod1, img_mod2 = self.img_mod(vec)  
        img_in = self.img_norm1(img)
        img_in = (1 + img_mod1.scale) * img_in + img_mod1.shift

        # 2) get q,k,v for image
        img_qkv = self.img_qkv(img_in)  # [B, L_img, 3*hidden_size]
        img_q, img_k, img_v = rearrange(img_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads)
        # QK-norm
        img_q, img_k = self.img_qk_norm(img_q, img_k, img_v)
        # apply 2D rope => shape [B, L_img, hidden_size]
        img_q, img_k = rotary_multiply_2d(img_q, img_k, img_rope)

        # 3) modulate text
        txt_mod1, txt_mod2 = self.txt_mod(vec)  
        txt_in = self.txt_norm1(txt)
        txt_in = (1 + txt_mod1.scale) * txt_in + txt_mod1.shift

        # 4) get q,k,v for text
        txt_qkv = self.txt_qkv(txt_in) 
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads)
        txt_q, txt_k = self.txt_qk_norm(txt_q, txt_k, txt_v)
        # apply 1D rope
        txt_q, txt_k = rotary_multiply_1d(txt_q, txt_k, txt_rope)

        # 5) joint attention
        # Cat along the L dimension
        q = torch.cat([txt_q, img_q], dim=2)  # [B,H, L_txt+L_img, D]
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Standard scaled dot-product attention:
        attn_out = attention(q, k, v, self.dropout)  # shape [B, L_txt+L_img, D]
        txt_attn = attn_out[:, : txt.shape[1]]  # first L_txt
        img_attn = attn_out[:, txt.shape[1] :]

        # 6) final projections
        #   - For image
        img_out = img + img_mod1.gate * self.img_proj(img_attn)
        # MoE for image
        img_out = img_out + img_mod2.gate * self.img_moe(
            (1 + img_mod2.scale) * self.img_norm2(img_out) + img_mod2.shift
        )

        #   - For text
        txt_out = txt + txt_mod1.gate * self.txt_proj(txt_attn)
        # MoE for text
        txt_out = txt_out + txt_mod2.gate * self.txt_moe(
            (1 + txt_mod2.scale) * self.txt_norm2(txt_out) + txt_mod2.shift
        )

        return img_out, txt_out
    
class SingleStreamBlock(nn.Module):
    """
    A DiT "single-stream" block with:
      - separate text/image QKV + RoPE
      - a single cross-attention pass over concatenated sequences
      - Sparse MoE in place of the original MLP
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        pretraining_tp: int = 2,
        num_shared_experts: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        # Modulation with "double=True" so we get two sets (mod1, mod2)
        #   - typically one set is used for the attn skip-connection
        #   - the second set is used for the MoE skip-connection
        self.modulation = Modulation(hidden_size, double=True)

        # First norm + linear for text
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.qk_norm = QKNorm(self.head_dim)

        # Output projection after attention
        self.proj = nn.Linear(hidden_size, hidden_size)

        # Second norm and sparse MoE, replacing what was originally an MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.moe = SparseMoeBlock(
            hidden_size,
            mlp_ratio,
            num_experts,
            num_experts_per_tok,
            pretraining_tp,
            num_shared_experts,
        )

        self.initialize_weights()

    def initialize_weights(self):
        """
        Simple init scheme similar to your DoubleStreamBlock.
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        self.moe.initialize_weights()

    def forward(
        self,
        img: torch.Tensor,   # [B, L_img, hidden_size]
        txt: torch.Tensor,   # [B, L_txt, hidden_size]
        vec: torch.Tensor,   # conditioning vector => for Modulation
        mask: torch.Tensor,  # optional mask for images (or None)
        h: int,              # image height (2D RoPE)
        w: int,              # image width  (2D RoPE)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L_txt, _ = txt.shape
        _, L_img, _ = img.shape

        # 1) Prepare text RoPE (1D)
        rope_txt = rope_1d(self.head_dim, L_txt)  # [L_txt, d]
        rope_txt = rope_txt.unsqueeze(0).repeat(B, 1, 1).to(txt.device)

        # 2) Prepare image RoPE (2D)
        rope_img = rope_2d(self.head_dim, h, w)   # [L_img, d], flattened from (h*w)
        rope_img = rope_img.unsqueeze(0).repeat(B, 1, 1).to(img.device)
        if mask is not None:
            rope_img = remove_masked_tokens(rope_img, mask)

        # 3) modulation parameters => "double=True" => mod1, mod2
        mod1, mod2 = self.modulation(vec)

        # ---- TEXT branch ----
        txt_in = self.norm(txt)
        txt_in = (1 + mod1.scale) * txt_in + mod1.shift
        txt_qkv = self.qkv(txt_in)  # [B, L_txt, 3*hidden_size]
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads
        )
        txt_q, txt_k = self.qk_norm(txt_q, txt_k, txt_v)
        # Apply 1D RoPE
        txt_q, txt_k = rotary_multiply_1d(txt_q, txt_k, rope_txt)

        # ---- IMAGE branch ----
        img_in = self.norm(img)
        img_in = (1 + mod1.scale) * img_in + mod1.shift
        img_qkv = self.qkv(img_in)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (three H D) -> three B H L D", three=3, H=self.num_heads
        )
        img_q, img_k = self.qk_norm(img_q, img_k, img_v)
        # Apply 2D RoPE
        img_q, img_k = rotary_multiply_2d(img_q, img_k, rope_img)

        # ---- Single-stream attention: concat txt + img ----
        q = torch.cat([txt_q, img_q], dim=2)  # [B, H, (L_txt+L_img), D]
        k = torch.cat([txt_k, img_k], dim=2)
        v = torch.cat([txt_v, img_v], dim=2)

        # Standard scaled dot-product attention across entire sequence
        attn_out = attention(q, k, v, dropout=self.dropout)
        # Split out txt/img
        txt_attn = attn_out[:, :L_txt]  # [B, L_txt, D]
        img_attn = attn_out[:, L_txt:]  # [B, L_img, D]

        # ---- Add first skip-connection + projection ----
        txt_out = txt + mod1.gate * self.proj(txt_attn)
        img_out = img + mod1.gate * self.proj(img_attn)

        # ---- Single MoE over entire sequence ----
        #   (We re-concat txt + img, run LN+shift+scale, pass to MoE, then split)
        x_out = torch.cat([txt_out, img_out], dim=1)  # [B, L_txt+L_img, hidden_size]
        x_in = self.norm2(x_out)
        x_in = (1 + mod2.scale) * x_in + mod2.shift

        x_moe = self.moe(x_in)
        x_out = x_out + mod2.gate * x_moe

        # final separation
        txt_out = x_out[:, :L_txt]
        img_out = x_out[:, L_txt:]

        return img_out, txt_out

## Rope helper functions

def rotary_multiply_1d(
    q: torch.Tensor, 
    k: torch.Tensor, 
    rope_1d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies standard 1D rotary embedding to (q,k).
    - q, k: shape [B, H, L, D]
    - rope_1d: shape [B, L, D], i.e. cos/sin interleaved
    Returns:
        (q_rotated, k_rotated) of the same shape.
    """
    B, H, L, D = q.shape

    # Reshape rope to [B, L, D/2, 2] => split into cos and sin
    rope_2 = rope_1d.view(B, L, D // 2, 2).float()

    q_2 = q.view(B, H, L, D // 2, 2).float()
    k_2 = k.view(B, H, L, D // 2, 2).float()

    cos_ = rope_2[..., 0].unsqueeze(1)  # => [B, 1, L, D//2]
    sin_ = rope_2[..., 1].unsqueeze(1)

    # standard rotation: (x, y) -> (x cos - y sin, x sin + y cos)
    x_q, y_q = q_2[..., 0], q_2[..., 1]
    x_k, y_k = k_2[..., 0], k_2[..., 1]

    q_rot = torch.stack([
        x_q * cos_ - y_q * sin_,
        x_q * sin_ + y_q * cos_
    ], dim=-1)
    k_rot = torch.stack([
        x_k * cos_ - y_k * sin_,
        x_k * sin_ + y_k * cos_
    ], dim=-1)

    # reshape back
    q_out = q_rot.view(B, H, L, D).type_as(q)
    k_out = k_rot.view(B, H, L, D).type_as(k)
    return q_out, k_out


def rotary_multiply_2d(
    q: torch.Tensor, 
    k: torch.Tensor, 
    rope_2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies 2D rotary embedding by splitting row vs. col halves.
    - q, k: shape [B, H, L, D]
    - rope_2d: shape [B, L, D], where the first half is row-emb, second half is col-emb
    Returns:
        (q_rotated, k_rotated) of the same shape.
    """
    B, H, L, D_total = q.shape
    half_dim = D_total // 2

    # The first half of rope_2d => row part
    # The second half => col part
    row_part = rope_2d[:, :, :half_dim]
    col_part = rope_2d[:, :, half_dim:]

    # Split q, k along their last dimension into row vs. col portions
    q_row, q_col = q.split(half_dim, dim=-1)
    k_row, k_col = k.split(half_dim, dim=-1)

    # Now rotate row half with row_part, col half with col_part
    q_row_out, k_row_out = rotary_multiply_1d(q_row, k_row, row_part)
    q_col_out, k_col_out = rotary_multiply_1d(q_col, k_col, col_part)

    # Concatenate the row-rotated and col-rotated
    q_out = torch.cat([q_row_out, q_col_out], dim=-1)
    k_out = torch.cat([k_row_out, k_col_out], dim=-1)
    return q_out, k_out