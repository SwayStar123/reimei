import math
import torch.nn as nn
from torch.nn.modules.normalization import RMSNorm
from transformer.math import rope_ids
from transformer.moe import MoeMLP
from .embed import EmbedND, PatchEmbed, TimestepEmbedder, MLPEmbedder, OutputLayer
from .utils import remove_masked_tokens, add_masked_tokens, unpatchify
from .backbone import BackboneParams, TransformerBackbone
from .token_mixer import TokenMixer, TokenMixerParameters
import torch
# from config import AE_SCALING_FACTOR, AE_SHIFT_FACTOR
from config import AE_SCALING_FACTORS
from dataclasses import dataclass

@dataclass
class ReiMeiParameters:
    use_mmdit: bool = True
    use_ec: bool = False
    use_moe: bool = False
    shared_mod: bool = False
    shared_attn_projs: bool = False,
    channels: int = 32
    patch_size: tuple[int, int] = (1,1)
    embed_dim: int = 1152
    num_layers: int = 24
    num_heads: int = 1152 // 64
    siglip_dim: int = 1152
    num_experts: int = 4
    capacity_factor: float = 2.0
    shared_experts: int = 2
    dropout: float = 0.1
    token_mixer_layers: int = 2
    image_text_expert_ratio: int = 4
    # m_d: float = 1.0

class ReiMei(nn.Module):
    """
    ReiMei is a image diffusion transformer model.

        Args:
        channels (int): Number of input channels in the image data.
        patch_size (Tuple[int, int]): Size of the patch.
        embed_dim (int): Dimension of the embedding space.
        num_layers (int): Number of layers in the transformer backbone.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        mlp_dim (int): Dimension of the multi-layer perceptron.
        text_embed_dim (int): Dimension of the text embedding.
        vector_embed_dim (int): Dimension of the vector embedding.
        num_experts (int, optional): Number of experts in the transformer backbone. Default is 4.
        capacity_factor (float, optional): Average number of experts per token. Default is 2.0.
        shared_experts (int, optional): Number of shared experts in the transformer backbone. Default is 2.
        dropout (float, optional): Dropout rate. Default is 0.1.
        patch_mixer_layers (int, optional): Number of layers in the patch mixer. Default is 2.

    Attributes:
        embed_dim (int): Dimension of the embedding space.
        channels (int): Number of input channels in the image data.
        time_embedder (TimestepEmbedder): Timestep embedding layer.
        image_embedder (MLPEmbedder): Image embedding layer.
        text_embedder (MLPEmbedder): Text embedding layer.
        vector_embedder (MLPEmbedder): Vector embedding layer.
        token_mixer (TokenMixer): Token mixer layer.
        backbone (TransformerBackbone): Transformer backbone model.
        output (MLPEmbedder): Output layer.
    """
    def __init__(self, params: ReiMeiParameters):
        super().__init__()
        self.params = params
        self.embed_dim = params.embed_dim
        self.head_dim = params.embed_dim // params.num_heads
        self.channels = params.channels
        self.patch_size = params.patch_size
        self.use_mmdit = params.use_mmdit
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(self.embed_dim)

        # Image embedding
        self.image_embedder = PatchEmbed(self.channels, self.embed_dim, self.patch_size)
        
        # Text embedding
        self.siglip_norm = nn.LayerNorm(params.siglip_dim)
        self.siglip_embedder = MLPEmbedder(params.siglip_dim, self.embed_dim, hidden_dim=self.embed_dim, num_layers=1)

        # Vector (y) embedding
        self.vec_norm = nn.LayerNorm(params.siglip_dim)
        self.vector_embedder = MLPEmbedder(params.siglip_dim, self.embed_dim, hidden_dim=self.embed_dim, num_layers=1)

        self.rope_embedder = EmbedND(dim=self.head_dim)

        # TokenMixer
        if params.token_mixer_layers > 0:
            self.use_token_mixer = True
            token_mixer_params = TokenMixerParameters(
                use_mmdit=params.use_mmdit,
                use_ec=params.use_ec,
                # use_moe=params.use_moe,
                shared_mod=params.shared_mod,
                use_moe=False,
                embed_dim=self.embed_dim,
                num_heads=params.num_heads,
                num_layers=params.token_mixer_layers,
                num_experts=params.num_experts,
                capacity_factor=params.capacity_factor,
                num_shared_experts=params.shared_experts,
                exp_ratio=params.image_text_expert_ratio,
                dropout=params.dropout,
            )
            self.token_mixer = TokenMixer(token_mixer_params)
        else:
            self.use_token_mixer = False

        # Backbone transformer model
        backbone_params = BackboneParams(
            use_mmdit=params.use_mmdit,
            use_ec=params.use_ec,
            use_moe=params.use_moe,
            shared_mod=params.shared_mod,
            shared_attn_projs=params.shared_attn_projs,
            embed_dim=self.embed_dim,
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            num_experts=params.num_experts,
            capacity_factor=params.capacity_factor,
            shared_experts=params.shared_experts,
            dropout=params.dropout,
            image_text_expert_ratio=params.image_text_expert_ratio,
        )
        self.backbone = TransformerBackbone(backbone_params)
        
        self.output_layer = OutputLayer(self.embed_dim, self.channels * self.patch_size[0] * self.patch_size[1])

        self.initialize_weights()

    def initialize_weights(self):
        s = 1.0 / math.sqrt(self.embed_dim)

        # # Initialize all linear layers and biases
        # def _basic_init(module):
        #     if isinstance(module, nn.LayerNorm):
        #         if module.weight is not None:
        #             nn.init.constant_(module.weight, 1.0)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.Linear):
        #         nn.init.normal_(module.weight, std=s)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)

        # # Initialize all linear layers and biases
        # def _basic_init(module):
        #     if isinstance(module, nn.Linear):
        #         nn.init.xavier_uniform_(module.weight)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.Conv2d):
        #         # Initialize convolutional layers like linear layers
        #         nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)
        #     elif isinstance(module, nn.LayerNorm):
        #         # Initialize LayerNorm layers
        #         if module.weight is not None:
        #             nn.init.constant_(module.weight, 1.0)
        #         if module.bias is not None:
        #             nn.init.constant_(module.bias, 0)


        # # Apply basic initialization to all modules
        # self.apply(_basic_init)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output_layer.mlp.mlp[-1].weight, 0)
        nn.init.constant_(self.output_layer.mlp.mlp[-1].bias, 0)

    def forward(self, img, time, sig_txt, sig_vec, img_mask=None, txt_mask=None):
        # img: (batch_size, channels, height, width)
        # time: (batch_size, 1)
        # sig_txt: (batch_size, seq_len, siglip_dim)
        # sig_vec: (batch_size, siglip_dim)
        # mask: (batch_size, num_tokens)
        batch_size, channels, height, width = img.shape
        ps_h, ps_w = self.patch_size
        patched_h, patched_w = height // ps_h, width // ps_w

        # Text embeddings
        sig_txt = self.siglip_norm(sig_txt)
        sig_txt = self.siglip_embedder(sig_txt)
        txt = sig_txt

        _, seq_len, _ = txt.shape

        # Vector embedding (timestep + vector_embeddings)
        time = self.time_embedder(time)
        sig_vec = self.vec_norm(sig_vec)
        vec = self.vector_embedder(sig_vec) + time

        # Image embedding
        img = self.image_embedder(img)

        rope_id = rope_ids(batch_size, patched_h, patched_w, seq_len, img.device, img.dtype)
        rope_pe = self.rope_embedder(torch.cat(rope_id, dim=1))

        # Token-mixer
        if self.use_token_mixer:
            img, txt = self.token_mixer(img, txt, vec, rope_pe)

        # Remove masked patches
        txt_rope_id, img_rope_id = rope_id
        if img_mask is not None:
            img = remove_masked_tokens(img, img_mask)
            img_rope_id = remove_masked_tokens(img_rope_id, img_mask)

        if txt_mask is not None:
            txt = remove_masked_tokens(txt, txt_mask)
            txt_rope_id = remove_masked_tokens(txt_rope_id, txt_mask)

        rope_pe = self.rope_embedder(torch.cat((txt_rope_id, img_rope_id), dim=1))

        # Backbone transformer model
        img = self.backbone(img, txt, vec, rope_pe)

        # Final output layer
        # (bs, unmasked_num_tokens, embed_dim) -> (bs, unmasked_num_tokens, in_channels)
        img = self.output_layer(img, vec)

        # Add masked patches
        if img_mask is not None:
            # (bs, unmasked_num_tokens, in_channels) -> (bs, num_tokens, in_channels)
            img = add_masked_tokens(img, img_mask)

        img = unpatchify(img, self.patch_size, height, width)
        
        return img
    
    @torch.no_grad()
    def sample(self, z, sig_emb, sig_vec, sample_steps=50, cfg=3.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device, torch.bfloat16).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        scaling_factors = torch.tensor(AE_SCALING_FACTORS).to(z.device, torch.bfloat16).view(1, -1, 1, 1)

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device, torch.bfloat16)

            vc = self(z, t, sig_emb, sig_vec, None, None).to(torch.bfloat16)
            if cfg != 1.0:
                null_sig_emb = torch.zeros(b, 1, self.params.siglip_dim).to(z.device, torch.bfloat16)
                null_sig_vec = torch.zeros(b, self.params.siglip_dim).to(z.device, torch.bfloat16)
                vu = self(z, t, null_sig_emb, null_sig_vec, None, None)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)

        return images[-1] / scaling_factors
