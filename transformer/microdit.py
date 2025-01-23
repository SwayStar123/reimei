import math
import torch.nn as nn
from .embed import sincos_2d, TimestepEmbedder, MLPEmbedder, OutputLayer
from .utils import remove_masked_tokens, add_masked_tokens
from .backbone import BackboneParams, TransformerBackbone
from .token_mixer import TokenMixer
import torch
from config import AE_SCALING_FACTOR
from dataclasses import dataclass

@dataclass
class ReiMeiParameters:
    channels: int
    embed_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    siglip_dim: int
    bert_dim: int
    num_experts: int = 4
    active_experts: int = 2
    shared_experts: int = 2
    dropout: float = 0.1
    token_mixer_layers: int = 2
    image_text_expert_ratio: int = 4
    m_d: float = 1.0

class ReiMei(nn.Module):
    """
    ReiMei is a image diffusion transformer model.

        Args:
        channels (int): Number of input channels in the image data.
        embed_dim (int): Dimension of the embedding space.
        num_layers (int): Number of layers in the transformer backbone.
        num_heads (int): Number of attention heads in the multi-head attention mechanism.
        mlp_dim (int): Dimension of the multi-layer perceptron.
        text_embed_dim (int): Dimension of the text embedding.
        vector_embed_dim (int): Dimension of the vector embedding.
        num_experts (int, optional): Number of experts in the transformer backbone. Default is 4.
        active_experts (int, optional): Number of active experts in the transformer backbone. Default is 2.
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
        
        self.embed_dim = params.embed_dim
        self.pos_emb_dim = params.embed_dim // params.num_heads
        self.channels = params.channels
        
        # Timestep embedding
        self.time_embedder = TimestepEmbedder(self.embed_dim)

        # Image embedding
        self.image_embedder = MLPEmbedder(self.channels, self.embed_dim)
        
        # Text embedding
        self.siglip_embedder = MLPEmbedder(params.siglip_dim, self.embed_dim)
        self.bert_embedder = MLPEmbedder(params.bert_dim, self.embed_dim)

        # Vector (y) embedding
        self.vector_embedder = MLPEmbedder(params.siglip_dim + params.bert_dim, self.embed_dim)
        
        # TokenMixer
        self.token_mixer = TokenMixer(self.embed_dim, params.num_heads, params.token_mixer_layers, num_experts=params.num_experts, num_experts_per_tok=params.active_experts)

        backbone_params = BackboneParams(
            input_dim=self.channels,
            embed_dim=self.embed_dim,
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            mlp_dim=params.mlp_dim,
            num_experts=params.num_experts,
            active_experts=params.active_experts,
            shared_experts=params.shared_experts,
            dropout=params.dropout,
            image_text_expert_ratio=params.image_text_expert_ratio,
        )

        # Backbone transformer model
        self.backbone = TransformerBackbone(backbone_params)
        
        self.output_layer = OutputLayer(self.embed_dim, self.channels)

        self.initialize_weights(params.m_d)

    def initialize_weights(self, m_d):
        # Initialize all linear layers and biases
        def _basic_init(module):
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _mup_init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02 / math.sqrt(m_d))

        # Apply basic initialization to all modules
        self.apply(_basic_init)
        self.apply(_mup_init)

        for embedder in [self.siglip_embedder, self.bert_embedder, self.vector_embedder, self.image_embedder]:
            nn.init.normal_(embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(embedder.mlp[2].weight, std=0.02)
            nn.init.constant_(embedder.mlp[0].bias, 0)
            nn.init.constant_(embedder.mlp[2].bias, 0)

        nn.init.normal_(self.time_embedder.mlp.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp.mlp[2].weight, std=0.02)
        nn.init.constant_(self.time_embedder.mlp.mlp[0].bias, 0)
        nn.init.constant_(self.time_embedder.mlp.mlp[2].bias, 0)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output_layer.mlp.weight, 0)
        nn.init.constant_(self.output_layer.mlp.bias, 0)

    def forward(self, img, time, sig_txt, sig_vec, bert_txt, bert_vec, mask=None):
        # img: (batch_size, channels, height, width)
        # time: (batch_size, 1)
        # sig_txt: (batch_size, seq_len, siglip_dim)
        # sig_vec: (batch_size, siglip_dim)
        # bert_txt: (batch_size, seq_len, bert_dim)
        # bert_vec: (batch_size, bert_dim)
        # mask: (batch_size, num_tokens)
        batch_size, channels, height, width = img.shape

        # Reshape and transmute img to have shape (batch_size, height*width, channels)
        img = img.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)

        # Text embeddings
        sig_txt = self.siglip_embedder(sig_txt)
        bert_txt = self.bert_embedder(bert_txt)
        txt = torch.cat([sig_txt, bert_txt], dim=1)
        
        # Vector embedding (timestep + vector_embeddings)
        time = self.time_embedder(time)

        vec = torch.cat([sig_vec, bert_vec], dim=1)
        vec = self.vector_embedder(vec) + time  # (batch_size, embed_dim)

        # Image embedding
        img = self.image_embedder(img)

        # (height, width, embed_dim)
        sincos_pos_embed = sincos_2d(self.embed_dim, height, width)
        sincos_pos_embed = sincos_pos_embed.to(img.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        img = img + sincos_pos_embed

        # Patch-mixer
        img, txt = self.token_mixer(img, txt, vec, height, width)

        # Remove masked patches
        if mask is not None:
            img = remove_masked_tokens(img, mask)

        # Backbone transformer model
        img = self.backbone(img, txt, vec, mask, height, width)
        
        # Final output layer
        # (bs, unmasked_num_tokens, embed_dim) -> (bs, unmasked_num_tokens, in_channels)
        img = self.output_layer(img, vec)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_tokens, in_channels) -> (bs, num_tokens, in_channels)
            img = add_masked_tokens(img, mask)

        img = img.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width).contiguous()
        
        return img
    
    @torch.no_grad()
    def sample(model, z, sig_txt, sig_vec, bert_txt, bert_vec, null_cond=None, sample_steps=2, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device, torch.bfloat16).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device, torch.bfloat16)

            vc = model(z, t, sig_txt, sig_vec, bert_txt, bert_vec, None).to(torch.bfloat16)
            # if null_cond is not None:
            #     vu = model(z, t, null_cond)
            #     vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)

        return (images[-1] / AE_SCALING_FACTOR)