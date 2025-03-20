from copy import deepcopy
import json
import random
import torch
from dataset.shapebatching_dataset import ShapeBatchingDataset, get_dataset
# from dataset.inet96 import ImageNet96Dataset
# from dataset.inet8bit import ImageNetDataset, InfiniteDataLoader
from transformer.reimei import ReiMei, ReiMeiParameters
from transformer.discriminator import Discriminator, DiscriminatorParameters, gan_loss_with_approximate_penalties
from accelerate import Accelerator
from config import AE_SCALING_FACTORS, AE_SHIFT_FACTOR, BS, CFG_RATIO, MAX_CAPTION_LEN, TRAIN_STEPS, MASK_RATIO, AE_CHANNELS, AE_HF_NAME, MODELS_DIR_BASE, SEED, SIGLIP_EMBED_DIM, DATASET_NAME, LR
from config import DIT_S as DIT
from datasets import load_dataset
from transformer.utils import expand_mask, random_cfg_mask, random_mask, apply_mask_to_tensor, remove_masked_tokens
from tqdm import tqdm
import datasets
import torchvision
import os
import pickle
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset
from diffusers import AutoencoderDC
from diffusers import AutoencoderKL
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR, LinearLR
import wandb
from transformers import SiglipModel, SiglipProcessor
import lovely_tensors
lovely_tensors.monkey_patch()

DTYPE = torch.bfloat16

@torch.no_grad
def sample_images(model, vae, noise, sig_emb, sig_vec):
    def normalize_batch(images):
        min_vals = images.amin(dim=(1, 2, 3), keepdim=True)  # Min per image
        max_vals = images.amax(dim=(1, 2, 3), keepdim=True)  # Max per image
        
        # Ensure no division by zero
        scale = (max_vals - min_vals).clamp(min=1e-8)
        
        return (images - min_vals) / scale

    # Use the stored embeddings
    # one_sampled_latents = model.module.sample(noise, sig_emb, sig_vec, sample_steps=1, cfg=5.0).to(device, dtype=DTYPE)
    # two_sampled_latents = model.module.sample(noise, sig_emb, sig_vec, sample_steps=2, cfg=5.0).to(device, dtype=DTYPE)
    # four_sampled_latents = model.module.sample(noise, sig_emb, sig_vec, sample_steps=4, cfg=5.0).to(device, dtype=DTYPE)
    fifty_sampled_latents = model.module.sample(noise, sig_emb, sig_vec, sample_steps=50, cfg=5.0).to(device, dtype=DTYPE)
    
    # Decode latents to images
    # one_sampled_images = normalize_batch(vae.decode(one_sampled_latents).sample)
    # two_sampled_images = normalize_batch(vae.decode(two_sampled_latents).sample)
    # four_sampled_images = normalize_batch(vae.decode(four_sampled_latents).sample)
    fifty_sampled_images = normalize_batch(vae.decode(fifty_sampled_latents).sample.clamp(-1, 1))

    # Log the sampled images
    # interleaved = torch.stack([one_sampled_images, two_sampled_images, four_sampled_images, fifty_sampled_images], dim=1).reshape(-1, *one_sampled_images.shape[1:])

    grid = torchvision.utils.make_grid(fifty_sampled_images, nrow=4, normalize=True, scale_each=True)

    return grid

def log_weight_norms(model, step):
    # Group parameters by component type
    embedder_norms = []
    mixer_norms = []
    backbone_norms = []
    other_norms = []
    
    # Track min, max, and mean across all parameters
    all_norms = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            norm = param.norm(2).item()
            all_norms.append(norm)
            
            # Group by component type based on name
            if 'embedder' in name.lower():
                embedder_norms.append(norm)
            elif 'mixer' in name.lower():
                mixer_norms.append(norm)
            elif 'backbone' in name.lower():
                backbone_norms.append(norm)
            else:
                other_norms.append(norm)
    
    # Calculate statistics
    norm_stats = {
        "weight_norm/mean": sum(all_norms) / len(all_norms) if all_norms else 0,
        "weight_norm/max": max(all_norms) if all_norms else 0,
        "weight_norm/min": min(all_norms) if all_norms else 0,
        "weight_norm/embedder_mean": sum(embedder_norms) / len(embedder_norms) if embedder_norms else 0,
        "weight_norm/mixer_mean": sum(mixer_norms) / len(mixer_norms) if mixer_norms else 0,
        "weight_norm/backbone_mean": sum(backbone_norms) / len(backbone_norms) if backbone_norms else 0,
        "weight_norm/other_mean": sum(other_norms) / len(other_norms) if other_norms else 0,
    }
    
    # Log the consolidated statistics
    wandb.log(norm_stats, step=step)

def log_grad_norms(model, step):
    # Group gradients by component type
    embedder_norms = []
    mixer_norms = []
    backbone_norms = []
    other_norms = []
    
    # Track min, max, and mean across all gradients
    all_norms = []
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norm = param.grad.norm(2).item()
            all_norms.append(norm)
            
            # Group by component type based on name
            if 'embedder' in name.lower():
                embedder_norms.append(norm)
            elif 'mixer' in name.lower():
                mixer_norms.append(norm)
            elif 'backbone' in name.lower():
                backbone_norms.append(norm)
            else:
                other_norms.append(norm)
    
    # Calculate statistics
    grad_stats = {
        "grad_norm/mean": sum(all_norms) / len(all_norms) if all_norms else 0,
        "grad_norm/max": max(all_norms) if all_norms else 0,
        "grad_norm/min": min(all_norms) if all_norms else 0,
        "grad_norm/embedder_mean": sum(embedder_norms) / len(embedder_norms) if embedder_norms else 0,
        "grad_norm/mixer_mean": sum(mixer_norms) / len(mixer_norms) if mixer_norms else 0,
        "grad_norm/backbone_mean": sum(backbone_norms) / len(backbone_norms) if backbone_norms else 0,
        "grad_norm/other_mean": sum(other_norms) / len(other_norms) if other_norms else 0,
    }
    
    # Log the consolidated statistics to wandb
    wandb.log(grad_stats, step=step)

if __name__ == "__main__":
    # Comment this out if you havent downloaded dataset and models yet
    # datasets.config.HF_HUB_OFFLINE = 1
    # torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "true"

    embed_dim = 3072
    patch_size = (1,1)

    params = ReiMeiParameters(
        use_mmdit=True,
        use_ec=True,
        use_moe=None,
        shared_mod=True,
        shared_attn_projs=True,
        channels=AE_CHANNELS,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=24,
        num_heads=(embed_dim // 128),
        siglip_dim=SIGLIP_EMBED_DIM,
        num_experts=4,
        capacity_factor=2.0,
        shared_experts=1,
        dropout=0.1,
        token_mixer_layers=1,
        image_text_expert_ratio=2,
    )

    accelerator = Accelerator(step_scheduler_with_optimizer=False)
    device = accelerator.device

    model = ReiMei(params)
    # model = torch.compile(ReiMei(params))

    params_count = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", params_count)

    if accelerator.is_main_process:
        wandb.init(project="ReiMei", config={
            "params_count": params_count,
            "dataset_name": DATASET_NAME,
            "ae_hf_name": AE_HF_NAME,
            "lr": LR,
            "bs": BS,
            "CFG_RATIO": CFG_RATIO,
            "MASK_RATIO": MASK_RATIO,
            "MAX_CAPTION_LEN": MAX_CAPTION_LEN,
            "params": params,
        })
    
    ds = get_dataset(BS, SEED + accelerator.process_index, device=device, dtype=DTYPE, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=LR, weight_decay=0.1)

    # scheduler = OneCycleLR(optimizer, max_lr=LR, total_steps=TRAIN_STEPS)
    # scheduler = ExponentialLR(optimizer, 0.9999995)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=TRAIN_STEPS)

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model, optimizer, scheduler, ds)
    # model, optimizer, ds = accelerator.prepare(model, optimizer, ds)

    # checkpoint = torch.load(f"models/pretrained_reimei_model_and_optimizer.pt")
    # checkpoint = torch.load(f"models/reimei_model_and_optimizer_3_f32.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # del checkpoint
    
    if accelerator.is_main_process:
        if "dc-ae" in AE_HF_NAME:
            ae = AutoencoderDC.from_pretrained(f"mit-han-lab/{AE_HF_NAME}", torch_dtype=DTYPE, cache_dir=f"{MODELS_DIR_BASE}/dc_ae", revision="main").to(device).eval()
        else:
            ae =  AutoencoderKL.from_pretrained(f"{AE_HF_NAME}", cache_dir=f"{MODELS_DIR_BASE}/vae").to(device=device, dtype=DTYPE).eval()
        # assert ae.config.scaling_factor == AE_SCALING_FACTOR, f"Scaling factor mismatch: {ae.config.scaling_factor} != {AE_SCALING_FACTOR}"
        
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        noise = torch.randn(12, AE_CHANNELS, 4, 4).to(device, dtype=DTYPE)
        example_batch = next(iter(ds))

        # example_latents = example_batch.to(device, dtype=DTYPE)[:4]

        example_latents = example_batch["ae_latent"][:12].to(device, dtype=DTYPE)
        # ex_sig_emb = example_batch["siglip_emb"][:4].to(device, dtype=DTYPE)
        # ex_sig_vec = example_batch["siglip_vec"][:4].to(device, dtype=DTYPE)

        example_captions = example_batch["caption"][:12]
        
        with torch.no_grad():
            example_ground_truth = ae.decode(example_latents).sample
        grid = torchvision.utils.make_grid(example_ground_truth, nrow=2, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid, f"logs/example_images.png")

        # Save captions
        with open("logs/example_captions.txt", "w") as f:
            for index, caption in enumerate(example_captions):
                f.write(f"{index}: {caption}\n")

        del grid, example_ground_truth, example_latents

        ex_captions = ["a cheeseburger on a white plate", "a bunch of bananas on a wooden table", "a white tea pot on a wooden table", "an erupting volcano with lava pouring out", "the aurora borealis above a snow covered forest", "bright blue sky with clouds", "a red apple on a wooden table", "a field of green grass with a snowcapped mountain in the background", "a blonde girl with her dog on the beach", "a red car and a blue car waiting at a red light signal", "a sleeping black cat by a window", "an astronaut riding a horse"]
        ex_sig_emb, ex_sig_vec = ds.encode_siglip(ex_captions)

        ae = ae.to("cpu")

    print("Starting training...")
    scaling_factors = torch.tensor(AE_SCALING_FACTORS).to(device, torch.bfloat16).view(1, -1, 1, 1)
    progress_bar = tqdm(ds, leave=False, total=TRAIN_STEPS)
    for batch_idx, batch in enumerate(progress_bar):
        # latents = batch_to_tensors(batch).to(device, DTYPE)
        # latents = batch.to(device, dtype=DTYPE)

        latents = batch["ae_latent"].to(device, dtype=DTYPE)

        bs, c, h, w = latents.shape
        latents = latents * scaling_factors

        siglip_emb = batch["siglip_emb"].to(device, dtype=DTYPE)
        siglip_vec = batch["siglip_vec"].to(device, dtype=DTYPE)

        img_mask = random_mask(bs, latents.shape[-2], latents.shape[-1], patch_size, mask_ratio=0.).to(device, dtype=DTYPE)
        cfg_mask = random_cfg_mask(bs, CFG_RATIO).to(device, dtype=DTYPE)

        siglip_emb = siglip_emb.to(device, dtype=DTYPE) * cfg_mask.view(bs, 1, 1)
        siglip_vec = siglip_vec.to(device, dtype=DTYPE) * cfg_mask.view(bs, 1)

        txt_mask = random_mask(bs, siglip_emb.size(1), 1, (1, 1), mask_ratio=MASK_RATIO).to(device=device, dtype=DTYPE)

        z = torch.randn_like(latents, device=device, dtype=DTYPE)

        nt = torch.randn((bs,), device=device, dtype=DTYPE)
        t = torch.sigmoid(nt)
        texp = t.view([bs, 1, 1, 1]).to(device, dtype=DTYPE)

        # correction_factor = 1 / torch.sqrt(1 - 2*texp*(1-texp))
        x_t = (1 - texp) * latents + texp * z

        vtheta = model(x_t, t, siglip_emb, siglip_vec, img_mask, txt_mask)

        img_mask = expand_mask(img_mask, latents.shape[-2], latents.shape[-1], patch_size)

        # Reshape from (BS, C, H, W) to (BS, H*W, C)
        vtheta_h = vtheta.permute(0, 2, 3, 1).reshape(bs, -1, AE_CHANNELS)
        latents_h = latents.permute(0, 2, 3, 1).reshape(bs, -1, AE_CHANNELS)
        z_h = z.permute(0, 2, 3, 1).reshape(bs, -1, AE_CHANNELS)

        vtheta_h = remove_masked_tokens(vtheta_h, img_mask)
        latents_h = remove_masked_tokens(latents_h, img_mask)
        z_h = remove_masked_tokens(z_h, img_mask)

        v = (z_h - latents_h)

        mse = (((v - vtheta_h) ** 2)).mean(dim=(1,2))

        loss = mse.mean()

        optimizer.zero_grad()
        accelerator.backward(loss)

        if batch_idx % 200 == 0 and accelerator.is_main_process:
            with torch.no_grad():
                log_grad_norms(model, batch_idx)

        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        progress_bar.set_postfix(loss=loss.item(), lr=current_lr)
        
        if accelerator.is_main_process:
            wandb.log({"loss": loss.item()}, step=batch_idx)

        del mse, loss, v, vtheta, latents, vtheta_h, latents_h, z_h, siglip_emb, siglip_vec

        if batch_idx % 200 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                with torch.no_grad():
                    log_weight_norms(model, batch_idx)
                    model.eval()
                    ae = ae.to(device)

                    grid = sample_images(model, ae, noise, ex_sig_emb, ex_sig_vec)
                    torchvision.utils.save_image(grid, f"logs/sampled_images_step_{batch_idx}.png")

                    del grid
                    
                    ae = ae.to("cpu")

                    model.train()

        if ((batch_idx % (TRAIN_STEPS//30)) == 0) and batch_idx != 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                model_save_path = f"models/reimei_model_and_optimizer_{batch_idx//(TRAIN_STEPS//30)}_f32.pt"
                torch.save({
                    'global_step': batch_idx,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, model_save_path)
                print(f"Model saved to {model_save_path}.")
        
        if batch_idx == TRAIN_STEPS - 1:
            print("Training complete.")

            # Save model in /models
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_optimizer = accelerator.unwrap_model(optimizer)
                model_save_path = "models/pretrained_reimei_model_and_optimizer.pt"
                torch.save(
                    {
                        'global_step': batch_idx,
                        'model_state_dict': unwrapped_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    },
                    model_save_path,
                )
                print(f"Model saved to {model_save_path}.")

            break