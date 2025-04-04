### 黎明 (reimei/límíng)
##### Meanings
1. daybreak; dawn;
2. dawn (of a new age)

---

Training and inference code for reimei, a diffusion model for image generation.


Model features
- MoE-MMDiT Blocks (SD3/Flux + DiT-MoE/EC-DiT)
- DC-AE autoencoder f64c128. IE trained on a highly compressed latent space with 128 channels
- SigLip text encoder
- Deferred masking of text tokens to reduce seq len in transformer during train time (256x256 res = 4x4 latent tokens, and 64 text tokens get masked down to 16 tokens. Total 32 tokens at train time. During 1024x1024 res finetuning, masking is removed)
- Shared parameters across layers, shares the AdaLN modulation weights, and QKVO projections for attention. (DiT-Air)
- Layerwise scaling of MLPs of transformer blocks

Find pretrained weights [here](https://huggingface.co/SwayStar123/ReiMei)

To run training
```pip install -r requirements.txt```
setup accelerate using 
```accelerate config```
then
```accelerate launch```
