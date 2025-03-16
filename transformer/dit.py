from dataclasses import dataclass
from einops import rearrange
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import RMSNorm

from transformer.math import attention
from transformer.moe import EC_SparseMoeBlock, TC_SparseMoeBlock

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

    def modulate(self, x):
        return (1 + self.scale) * x + self.shift

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.linear(self.silu(vec)[:, None, :]).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )
    
class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, dropout=(self.dropout if self.training else 0.0))
        x = self.proj(x)
        return x

#################################################################################
#                                 Core DiT Modules                              #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_dim,
        num_experts=8, 
        num_experts_per_tok=2, 
        pretraining_tp=2, 
        num_shared_experts=2, 
        use_moe: bool = False, 
        use_expert_choice: bool = False, 
        dropout=0.1,
        shared_mod=False,
        shared_attn_projs=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if not shared_attn_projs:
            self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if use_moe:
            if use_expert_choice:
                self.mlp = EC_SparseMoeBlock(hidden_size, mlp_dim, num_experts, float(num_experts_per_tok), pretraining_tp, num_shared_experts=num_shared_experts)
            else:
                self.mlp = TC_SparseMoeBlock(hidden_size, mlp_dim, num_experts, int(num_experts_per_tok), pretraining_tp, num_shared_experts=num_shared_experts)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_dim, hidden_size, bias=True),
            )

        if not shared_mod:
            self.mod = Modulation(hidden_size, True)

    def forward(self, x, c, pe, mod=None, attn=None):
        if mod is not None:
            msa, mlp = mod(c)
        else:
            msa, mlp = self.mod(c)

        if attn is not None:
            attn = attn
        else:
            attn = self.attn

        # x = x + msa.gate.unsqueeze(1) * self.attn(modulate(self.norm1(x).to(x.dtype), msa.shift, msa.scale))
        x = x + msa.gate * attn(msa.modulate(self.norm1(x)), pe)
        # x = x + mlp.gate.unsqueeze(1) * self.mlp(modulate(self.norm2(x).to(x.dtype), mlp.shift, mlp.scale))
        x = x + mlp.gate * self.mlp(mlp.modulate(self.norm2(x)))
        return x

class DoubleStreamBlock(nn.Module):
    """
    A DiT block with seperate MoE for text & image
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        num_experts=8,
        capacity_factor=2.0,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1,
        exp_ratio: int = 4,
        use_moe: bool = False,
        use_expert_choice: bool = False,
        shared_mod=False,
        shared_attn_projs=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        if not shared_mod:
            self.img_mod = Modulation(hidden_size, double=True)
            self.txt_mod = Modulation(hidden_size, double=True)
        
        self.shared_attn_projs = shared_attn_projs
        if not shared_attn_projs:
            self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, dropout=dropout)
            self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, dropout=dropout)

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        text_exps = max(1, num_experts // exp_ratio)
        if use_moe:
            if use_expert_choice:
                text_capacity = min(float(text_exps), float(capacity_factor))

                self.img_mlp = EC_SparseMoeBlock(
                    hidden_size, mlp_dim, num_experts, float(capacity_factor), pretraining_tp, num_shared_experts
                )
                self.txt_mlp = EC_SparseMoeBlock(
                    hidden_size, mlp_dim, text_exps, text_capacity, pretraining_tp, num_shared_experts
                )
            else:
                text_capacity = min(int(text_exps), int(capacity_factor))
                self.img_mlp = TC_SparseMoeBlock(
                    hidden_size, mlp_dim, num_experts, int(capacity_factor), pretraining_tp, num_shared_experts
                )
                self.txt_mlp = TC_SparseMoeBlock(
                    hidden_size, mlp_dim, text_exps, text_capacity, pretraining_tp, num_shared_experts
                )
        else:
            self.img_mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_dim, hidden_size, bias=True),
            )
            self.txt_mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_dim, hidden_size, bias=True),
            )

    def forward(
        self,
        img: Tensor,          # [B, L_img, hidden_size]
        txt: Tensor,          # [B, L_txt, hidden_size]
        vec: Tensor,          # conditioning vector => Modulation
        pe: Tensor,    # rope positional encoding
        mod=None,
        img_attn=None,
        txt_attn=None,
    ) -> tuple[Tensor, Tensor]:
        dtype = img.dtype
        if mod is not None:
            img_mod1, img_mod2 = mod(vec)
            txt_mod1, txt_mod2 = mod(vec)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        if self.shared_attn_projs:
            img_attn = img_attn
            txt_attn = txt_attn
        else:
            img_attn = self.img_attn
            txt_attn = self.txt_attn

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = img_mod1.modulate(img_modulated)
        img_qkv = img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = txt_mod1.modulate(txt_modulated)
        txt_qkv = txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn_out, img_attn_out = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * img_attn.proj(img_attn_out)
        img = img + img_mod2.gate * self.img_mlp((img_mod2.modulate(self.img_norm2(img))).to(dtype))

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * txt_attn.proj(txt_attn_out)
        txt = txt + txt_mod2.gate * self.txt_mlp((txt_mod2.modulate(self.txt_norm2(txt))).to(dtype))
        
        return img, txt

class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        num_experts=8,
        capacity_factor=2.0,
        pretraining_tp=2,
        num_shared_experts=2,
        dropout: float = 0.1,
        use_moe: bool = False,
        use_expert_choice: bool = False,
        shared_mod=False,
        shared_attn_projs=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        self.dropout = dropout

        # qkv and mlp_in
        self.shared_attn_projs = shared_attn_projs

        if not shared_attn_projs:
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3)

        if use_moe:
            if use_expert_choice:
                self.mlp = EC_SparseMoeBlock(
                    hidden_size, mlp_dim, num_experts, float(capacity_factor), pretraining_tp, num_shared_experts
                )
            else:
                self.mlp = TC_SparseMoeBlock(
                    hidden_size, mlp_dim, num_experts, int(capacity_factor), pretraining_tp, num_shared_experts
                )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, mlp_dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_dim, hidden_size, bias=True),
            )

        # proj and mlp_out
        if not shared_attn_projs:
            self.linear2 = nn.Linear(2 * hidden_size, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")

        if not shared_mod:
            self.modulation = Modulation(hidden_size, double=False)


    def forward(
        self,
        x: Tensor,   # [B, L_img + L_txt, hidden_size]
        vec: Tensor,   # conditioning vector => for Modulation
        pe: Tensor,    # rope positional encoding
        mod=None,
        linear1=None,
        linear2=None,
    ) -> tuple[Tensor, Tensor]:
        if mod is not None:
            mod1, _ = mod(vec)
        else:
            mod1, _ = self.mod(vec)

        if self.shared_attn_projs:
            linear1 = linear1
            linear2 = linear2
        else:
            linear1 = self.linear1
            linear2 = self.linear2

        x_mod = mod1.modulate(self.pre_norm(x))
        qkv = linear1(x_mod)
        mlp_out = self.mlp(x_mod)
        # qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        qkv = qkv.contiguous()
        mlp_out = mlp_out.contiguous()

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, dropout=(self.dropout if self.training else 0.0))
        # compute activation in mlp stream, cat again and run second linear layer
        output = linear2(torch.cat((attn, self.mlp_act(mlp_out)), 2))
        return x + mod1.gate * output
