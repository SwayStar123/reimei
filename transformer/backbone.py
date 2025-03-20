import torch
from transformer.dit import DoubleStreamBlock, Modulation, SelfAttention, SingleStreamBlock, DiTBlock
from torch import nn
from dataclasses import dataclass

@dataclass
class BackboneParams:
    use_mmdit: bool = True
    use_ec: bool = False
    use_moe: bool = False
    shared_mod: bool = False
    shared_attn_projs: bool = False
    embed_dim: int = 1152
    num_layers: int = 24
    num_heads: int = 1152 // 64
    num_experts: int = 4
    capacity_factor: float = 2.0
    shared_experts: int = 2
    dropout: float = 0.1
    image_text_expert_ratio: int = 4
    pretraining_tp: int = 1

def nearest_divisor(scaled_num_heads, embed_dim):
    # Find all divisors of embed_dim
    divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
    
    # Find the nearest divisor
    nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))
    
    return nearest

class TransformerBackbone(nn.Module):
    def __init__(self, params: BackboneParams):
        super().__init__()
        mf_min, mf_max = 1.0, 3.0
        self.use_mmdit = params.use_mmdit
        self.shared_mod = params.shared_mod
        if self.shared_mod:
            self.mod = Modulation(params.embed_dim, True)
        
        self.shared_attn_projs = params.shared_attn_projs
        if self.shared_attn_projs:
            if self.use_mmdit:
                self.img_attn = SelfAttention(params.embed_dim, params.num_heads, dropout=params.dropout)
                self.txt_attn = SelfAttention(params.embed_dim, params.num_heads, dropout=params.dropout)
            
            self.attn = SelfAttention(params.embed_dim, params.num_heads, dropout=params.dropout)

        self.double_layers = nn.ModuleList()
        self.layers = nn.ModuleList()

        for i in range(params.num_layers):
            # Calculate scaling factors for the i-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * i / (params.num_layers - 1)

            # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = (int(params.embed_dim * mf) // params.num_heads) * params.num_heads
            scaled_num_heads = params.num_heads
            mlp_dim = max(params.embed_dim, scaled_mlp_dim)

            if i % 2 == 0:  # Even layers use regular DiT (no MoE)
                n_exp = 1
                n_shared = None
                n_act = 1.0
            else:  # Odd layers use MoE DiT
                n_exp = params.num_experts
                n_shared = params.shared_experts
                n_act = min(params.capacity_factor, float(n_exp))

            if params.use_mmdit:
                if i < params.num_layers // 6: # First sixth uses DoubleStreamBlock
                    self.double_layers.append(DoubleStreamBlock(
                        hidden_size=params.embed_dim,
                        num_heads=scaled_num_heads,
                        mlp_dim=mlp_dim,
                        num_experts=n_exp,
                        capacity_factor=n_act,
                        pretraining_tp=params.pretraining_tp,
                        num_shared_experts=n_shared,
                        dropout=params.dropout,
                        exp_ratio=params.image_text_expert_ratio,
                        use_moe=params.use_moe,
                        use_expert_choice=params.use_ec,
                        shared_mod=params.shared_mod,
                        shared_attn_projs=params.shared_attn_projs
                    ))
                else:  # Rest use SingleStreamBlock
                    # self.layers.append(SingleStreamBlock(
                    #     hidden_size=params.embed_dim,
                    #     num_heads=scaled_num_heads,
                    #     mlp_dim=mlp_dim,
                    #     num_experts=n_exp,
                    #     capacity_factor=n_act,
                    #     pretraining_tp=params.pretraining_tp,
                    #     num_shared_experts=n_shared,
                    #     dropout=params.dropout,
                    #     use_moe=params.use_moe,
                    #     use_expert_choice=params.use_ec,
                    #     shared_mod=params.shared_mod,
                    #     shared_attn_projs=params.shared_attn_projs
                    # ))
                    self.layers.append(DiTBlock(
                        hidden_size=params.embed_dim,
                        num_heads=scaled_num_heads,
                        mlp_dim=mlp_dim,
                        num_experts=n_exp,
                        num_experts_per_tok=n_act,
                        pretraining_tp=params.pretraining_tp,
                        num_shared_experts=n_shared,
                        dropout=params.dropout,
                        use_moe=params.use_moe,
                        use_expert_choice=params.use_ec,
                        shared_mod=params.shared_mod,
                        shared_attn_projs=params.shared_attn_projs
                    ))
            else:
                self.layers.append(DiTBlock(
                    hidden_size=params.embed_dim,
                    num_heads=scaled_num_heads,
                    mlp_dim=mlp_dim,
                    num_experts=n_exp,
                    num_experts_per_tok=n_act,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=n_shared,
                    dropout=params.dropout,
                    use_moe=params.use_moe,
                    use_expert_choice=params.use_ec,
                    shared_mod=params.shared_mod,
                    shared_attn_projs=params.shared_attn_projs
                ))

    def forward(
            self, 
            img: torch.Tensor,
            txt: torch.Tensor,
            vec: torch.Tensor,
            pe: torch.Tensor,
            mod = None,
            img_attn = None,
            txt_attn = None,
            attn = None,
            ):
        if self.shared_mod:
            mod = self.mod

        if self.shared_attn_projs:
            if self.use_mmdit:
                img_attn = self.img_attn
                txt_attn = self.txt_attn
            attn = self.attn

        for layer in self.double_layers:
            img, txt = layer(img, txt, vec, pe, mod=mod, img_attn=img_attn, txt_attn=txt_attn)

        img = torch.cat((txt, img), 1)

        for layer in self.layers:
            img = layer(img, vec, pe, mod=mod, attn=attn)
        
        img = img[:, txt.shape[1]:, ...]

        return img