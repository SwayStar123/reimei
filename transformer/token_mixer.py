from dataclasses import dataclass
import torch
import torch.nn as nn
from transformer.dit import DoubleStreamBlock, DiTBlock, Modulation

@dataclass
class TokenMixerParameters:
    use_mmdit: bool = True
    use_ec: bool = False
    use_moe: bool = False
    shared_mod: bool = False
    embed_dim: int = 1152
    num_heads: int = 1152 // 64
    num_layers: int = 2
    mlp_ratio: int = 2
    num_experts: int = 8
    capacity_factor: int = 2.0
    pretraining_tp: int = 2
    num_shared_experts: int = 2
    exp_ratio: int = 4
    dropout: float = 0.1

class TokenMixer(nn.Module):
    """
    Each layer expects:
        - img:       [B, L_img, embed_dim]
        - txt:       [B, L_txt, embed_dim]
        - vec:       [B, embed_dim]            (conditioning vector for Modulation)
        - h          Height of the original image
        - w          Width of the original image
    and returns the updated (img, txt) after `num_layers` of DoubleStreamBlock.
    """
    def __init__(
        self,
        params: TokenMixerParameters,
    ):
        super().__init__()
        self.use_mmdit = params.use_mmdit
        self.shared_mod = params.shared_mod
        if self.shared_mod:
            self.mod = Modulation(params.embed_dim, True)

        if params.use_mmdit:
            self.layers = nn.ModuleList([
                DoubleStreamBlock(
                    hidden_size=params.embed_dim,
                    num_heads=params.num_heads,
                    mlp_dim=params.mlp_ratio * params.embed_dim,
                    num_experts=params.num_experts,
                    capacity_factor=params.capacity_factor,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=params.num_shared_experts,
                    exp_ratio=params.exp_ratio,
                    use_moe=params.use_moe,
                    use_expert_choice=params.use_ec,
                    shared_mod=params.shared_mod
                )
                for _ in range(params.num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                DiTBlock(
                    hidden_size=params.embed_dim,
                    num_heads=params.num_heads,
                    mlp_dim=params.mlp_ratio * params.embed_dim,
                    num_experts=params.num_experts,
                    num_experts_per_tok=params.capacity_factor,
                    pretraining_tp=params.pretraining_tp,
                    num_shared_experts=params.num_shared_experts,
                    use_moe=params.use_moe,
                    use_expert_choice=params.use_ec,
                    dropout=params.dropout
                )
                for _ in range(params.num_layers)
            ])

    def forward(
        self,
        img: torch.Tensor,       # [B, L_img, embed_dim]
        txt: torch.Tensor,       # [B, L_txt, embed_dim]
        vec: torch.Tensor,       # [B, embed_dim]
        pe: torch.Tensor,    # rope positional encoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shared_mod:
            mod = self.mod
        else:
            mod = None

        if self.use_mmdit:
            for layer in self.layers:
                img, txt = layer(img, txt, vec, pe, mod=mod)
        else:
            img = torch.cat((txt, img), 1)

            for layer in self.layers:
                img = layer(img, vec, pe, mod=mod)

            img = img[:, txt.shape[1]:, ...]
            txt = img[:, :txt.shape[1], ...]

        return img, txt
