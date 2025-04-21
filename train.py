from copy import deepcopy
import json
import random
import torch
from dataset.shapebatching_dataset import ShapeBatchingDataset, get_dataset
# from dataset.inet96 import ImageNet96Dataset
# from dataset.inet8bit import ImageNetDataset, InfiniteDataLoader
from transformer.reimei import ReiMei, ReiMeiParameters
from transformer.discriminator import Discriminator, DiscriminatorParameters, gan_loss_with_approximate_penalties
import deepspeed
import argparse
from config import (
    AE_SCALING_FACTORS, AE_SHIFT_FACTOR, BS, CFG_RATIO, MAX_CAPTION_LEN,
    TRAIN_STEPS, MASK_RATIO, AE_CHANNELS, AE_HF_NAME, MODELS_DIR_BASE,
    SEED, SIGLIP_EMBED_DIM, DATASET_NAME, LR, SIGLIP_HF_NAME
)
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
from transformers import SiglipModel, SiglipProcessor, SiglipTextModel
import lovely_tensors
import torch.multiprocessing as mp
lovely_tensors.monkey_patch()

DTYPE = torch.bfloat16

@torch.no_grad
def sample_images(model, vae, noise, sig_emb, sig_vec):
    fifty_sampled_latents = model.sample(noise, sig_emb, sig_vec, sample_steps=50, cfg=5.0).to(vae.device, dtype=DTYPE)
    
    fifty_sampled_images = vae.decode(fifty_sampled_latents).sample.clamp(-1, 1)

    grid = torchvision.utils.make_grid(fifty_sampled_images, nrow=4, normalize=True, scale_each=True, value_range=(-1, 1))

    return grid

def log_weight_norms(model, step):
    # Group parameters by component type
    embedder_norms = []
    mixer_norms = []
    backbone_norms = []
    other_norms = []
    
    # Track min, max, and mean across all parameters
    all_norms = []
    
    # Access parameters via model_engine.module if using deepspeed engine
    model_to_log = model.module if hasattr(model, 'module') else model
    
    for name, param in model_to_log.named_parameters():
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
    
    # Access parameters via model_engine.module if using deepspeed engine
    model_to_log = model.module if hasattr(model, 'module') else model
    
    for name, param in model_to_log.named_parameters():
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

def main():
    # --- Set Multiprocessing Start Method ---
    # Must be called early, ideally before any CUDA/multiprocessing stuff happens
    # Use try-except block as it can only be set once.
    try:
        mp.set_start_method('spawn', force=True)
        print("MP Start Method set to 'spawn'")
    except RuntimeError as e:
        if "context has already been set" not in str(e):
             print(f"Could not set MP start method: {e}")
        else:
             print("MP Start Method was already set to 'spawn'")
             pass # Ignore if already set
    # -----------------------------------------

    # --- Argument Parser for DeepSpeed ---
    parser = argparse.ArgumentParser(description='ReiMei Training with DeepSpeed')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # Add DeepSpeed arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    # -------------------------------------

    # Comment this out if you havent downloaded dataset and models yet
    # datasets.config.HF_HUB_OFFLINE = 1
    # torch.set_float32_matmul_precision('high')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "true"

    # --- DeepSpeed World Setup ---
    deepspeed.init_distributed()
    local_rank = args.local_rank
    # Check if local_rank is valid, otherwise default to 0 for single GPU case (though unlikely with deepspeed launcher)
    if local_rank == -1:
        local_rank = 0 
        # Consider adding a warning or specific handling if needed
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    global_rank = deepspeed.comm.get_rank()
    world_size = deepspeed.comm.get_world_size()
    # -----------------------------

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

    model = ReiMei(params)
    # --- Disable torch.compile ---
    # print(f"[Rank {global_rank}] Compiling model with torch.compile...")
    # model = torch.compile(model)
    # print(f"[Rank {global_rank}] Model compiled.")
    # ---------------------------

    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Rank {global_rank}] Number of trainable parameters: {params_count}")

    if global_rank == 0:
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
            "deepspeed_config": args.deepspeed_config
        })
    
    ds_seed = SEED + global_rank
    # Create the dataset instance first
    ds = get_dataset(BS // world_size, ds_seed, device, DTYPE, num_workers=0) # Pass device for siglip

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=LR, weight_decay=0.1)

    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=TRAIN_STEPS)

    # --- DeepSpeed Initialization ---
    # Initialize WITHOUT passing training_data
    model_engine, optimizer, _, scheduler = deepspeed.initialize( # Note: 3rd return value is dataloader, now assigned to _
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler
        # Removed: training_data=ds
    )

    # --- Manually Create DataLoader ---
    # Since ds is IterableDataset, do not specify sampler.
    # DataLoader will handle distributing work across ranks/workers for IterableDataset.
    # The batching is done inside ShapeBatchingDataset, so no collate_fn needed here.
    train_dataloader = DataLoader(
        ds,
        batch_size=1,
        num_workers=2,
        pin_memory=True
    )
    # ----------------------------------

    ae = None
    noise = None
    ex_sig_emb = None
    ex_sig_vec = None

    if global_rank == 0:
        vae_device = device
        if "dc-ae" in AE_HF_NAME:
            ae = AutoencoderDC.from_pretrained(f"mit-han-lab/{AE_HF_NAME}", torch_dtype=DTYPE, cache_dir=f"{MODELS_DIR_BASE}/dc_ae", revision="main").to(vae_device).eval()
        else:
            ae =  AutoencoderKL.from_pretrained(f"{AE_HF_NAME}", cache_dir=f"{MODELS_DIR_BASE}/vae").to(device=vae_device, dtype=DTYPE).eval()
        
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # --- Get and Prepare Example Batch ---
        example_batch = next(iter(train_dataloader))

        # Squeeze the outer dimension before slicing and moving to device
        example_latents_squeezed = example_batch["ae_latent"].squeeze(0) # -> (inner_bs, C, H, W)
        example_latents = example_latents_squeezed[:16].to(vae_device, dtype=DTYPE) # Slice the inner batch dim

        # Captions don't need squeezing if they are a list within the batch dict key
        # Ensure it's accessing the correct list structure if batch['caption'] is nested
        # Assuming batch['caption'] is already a list of strings of length `inner_bs` after squeeze? Check dataset output.
        # If batch['caption'] was also (1, inner_bs, ...), it might need squeezing too.
        # Let's assume the dataset puts a flat list in batch['caption'] for now.
        # If example_batch['caption'] has shape (1, inner_bs), squeeze it:
        example_captions_raw = example_batch["caption"]
        if isinstance(example_captions_raw, torch.Tensor) and example_captions_raw.shape[0] == 1:
             example_captions_squeezed = example_captions_raw.squeeze(0)
        elif isinstance(example_captions_raw, list) and len(example_captions_raw) == 1 and isinstance(example_captions_raw[0], list):
             # Handle potential nested list [[caption1, caption2,...]]
             example_captions_squeezed = example_captions_raw[0]
        else:
             example_captions_squeezed = example_captions_raw # Assume it's already the flat list

        example_captions = example_captions_squeezed[:16]
        # ------------------------------------
        
        print(f"[Rank 0] Decoding example latents with shape: {example_latents.shape}") # Add print to verify shape
        with torch.no_grad():
            # Decode should now work with the 4D tensor
            example_ground_truth = ae.decode(example_latents).sample
        grid = torchvision.utils.make_grid(example_ground_truth, nrow=2, normalize=True, scale_each=True)
        torchvision.utils.save_image(grid, f"logs/example_images.png")

        with open("logs/example_captions.txt", "w") as f:
            # Ensure we are iterating over the correct list of captions
            for index, caption in enumerate(example_captions):
                f.write(f"{index}: {caption}\n")

        del grid, example_ground_truth, example_latents # Keep squeezed variables if needed later? No, cleanup ok.

        ex_captions = ["a cheeseburger on a white plate", "a bunch of bananas on a wooden table", "a white tea pot on a wooden table", "an erupting volcano with lava pouring out", "the aurora borealis northern lights fill the starry night sky with a bright glow above a snow covered forest with mountains in the background", "a sunflower wearing sunglasses, cloudy blue sky background", "a red apple on a wooden table", "a field of green grass with a snowcapped mountain in the background", "a blonde girl chasing after her dog on a beach", "a red car and a blue car waiting at a red light signal", "a sleeping black cat by a window", "an astronaut riding a horse", "a smilling couple dressed in formal wedding attire", "The eiffel tower standing atop an icy glacier in the north american arctic", "An unlit fireplace with a TV above it. The TV shows a lion", "a black car in the middle of a beautiful endless field of white flowers."]
        
        siglip_processor = SiglipProcessor.from_pretrained(SIGLIP_HF_NAME)
        siglip_text_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(vae_device).eval()
        
        inputs = siglip_processor(text=ex_captions, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_CAPTION_LEN).to(vae_device)
        with torch.no_grad():
             outputs = siglip_text_model(**inputs, output_hidden_states=True)
             ex_sig_emb = outputs.hidden_states[-1].to(vae_device, dtype=DTYPE)
             if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                 ex_sig_vec = outputs.pooler_output.to(vae_device, dtype=DTYPE)
             else:
                 print("[Rank 0 Warning] SiglipTextModel output missing pooler_output, using CLS token embedding instead.")
                 ex_sig_vec = outputs.last_hidden_state[:, 0].to(vae_device, dtype=DTYPE)
        del siglip_processor, siglip_text_model, inputs, outputs

        noise = torch.randn(16, AE_CHANNELS, 4, 4).to(vae_device, dtype=DTYPE)

    print(f"[Rank {global_rank}] Starting training...")
    scaling_factors = torch.tensor(AE_SCALING_FACTORS).to(device, torch.bfloat16).view(1, -1, 1, 1)
    
    progress_bar = tqdm(train_dataloader, leave=False, total=TRAIN_STEPS, disable=(global_rank != 0))
    
    global_step = 0
    
    while global_step < TRAIN_STEPS:
        for batch_idx, batch in enumerate(progress_bar):
            if global_step >= TRAIN_STEPS:
                break
                
            # --- Squeeze outer dimension and move data to device ---
            # batch contains tensors like (1, inner_bs, C, H, W)
            latents = batch["ae_latent"].squeeze(0) # Remove outer dim: -> (inner_bs, C, H, W)
            siglip_emb = batch["siglip_emb"].squeeze(0) # -> (inner_bs, SeqLen, EmbDim)
            siglip_vec = batch["siglip_vec"].squeeze(0) # -> (inner_bs, EmbDim)

            # Move to device after squeezing
            latents = latents.to(device, dtype=DTYPE, non_blocking=True)
            siglip_emb = siglip_emb.to(device, dtype=DTYPE, non_blocking=True)
            siglip_vec = siglip_vec.to(device, dtype=DTYPE, non_blocking=True)
            # Captions are likely strings, shapes are tuples/ints - don't need moving

            # --- Correctly unpack dimensions AFTER squeezing ---
            bs_actual = latents.shape[0] # Now gets the inner batch size (e.g., 25)
            c, h, w = latents.shape[1], latents.shape[2], latents.shape[3] # c=128, h=3, w=5

            # Optional: Add print to confirm shapes after squeeze
            # if global_step == 0 and batch_idx == 0 and global_rank == 0:
            #    print(f"After Squeeze - Latents: {latents.shape}, SiglipEmb: {siglip_emb.shape}, SiglipVec: {siglip_vec.shape}")
            #    print(f"bs_actual={bs_actual}, c={c}, h={h}, w={w}")

            # --- The rest of the training loop ---
            # Ensure scaling_factors is on the correct device
            scaling_factors = scaling_factors.to(device, dtype=DTYPE) # Add this just in case

            # Perform multiplication - c should now be correct (128)
            latents = latents * scaling_factors[:, :c, :, :]

            # Create masks using bs_actual (inner batch size)
            img_mask = random_mask(bs_actual, h, w, patch_size, mask_ratio=0.).to(device, dtype=DTYPE) # Use correct h, w
            cfg_mask = random_cfg_mask(bs_actual, CFG_RATIO).to(device, dtype=DTYPE)

            # Apply cfg mask - siglip tensors already have correct bs_actual dim
            siglip_emb = siglip_emb * cfg_mask.view(bs_actual, 1, 1)
            siglip_vec = siglip_vec * cfg_mask.view(bs_actual, 1)

            txt_mask = random_mask(bs_actual, siglip_emb.size(1), 1, (1, 1), mask_ratio=MASK_RATIO).to(device=device, dtype=DTYPE)

            z = torch.randn_like(latents, device=device, dtype=DTYPE) # Use latents shape (inner_bs, C, H, W)

            nt = torch.randn((bs_actual,), device=device, dtype=DTYPE)
            t = torch.sigmoid(nt)
            texp = t.view([bs_actual, 1, 1, 1]).to(device, dtype=DTYPE)

            x_t = (1 - texp) * latents + texp * z

            # Pass squeezed siglip tensors to model
            vtheta = model_engine(x_t, t, siglip_emb, siglip_vec, img_mask, txt_mask)

            # Reshaping and masking logic should now use correct dimensions (h, w)
            img_mask_expanded = expand_mask(img_mask, h, w, patch_size) # Use correct h, w

            vtheta_h = vtheta.permute(0, 2, 3, 1).reshape(bs_actual, -1, c) # Use correct c
            latents_h = latents.permute(0, 2, 3, 1).reshape(bs_actual, -1, c) # Use correct c
            z_h = z.permute(0, 2, 3, 1).reshape(bs_actual, -1, c)             # Use correct c

            vtheta_h = remove_masked_tokens(vtheta_h, img_mask)
            latents_h = remove_masked_tokens(latents_h, img_mask)
            z_h = remove_masked_tokens(z_h, img_mask)

            v = z_h - latents_h
            mse = (((v - vtheta_h) ** 2)).mean(dim=(1,2))
            loss = mse.mean()

            model_engine.backward(loss)
            
            # Optional: Gradient Norm Logging (before optimizer step)
            # Increase frequency (e.g., every 1000 steps)
            if global_step % 1000 == 0 and global_rank == 0:
                 with torch.no_grad():
                      log_grad_norms(model_engine, global_step)

            model_engine.step() # Handles grad accum, clipping, optimizer step, scheduler step

            current_lr = optimizer.param_groups[0]['lr']

            if global_rank == 0:
                progress_bar.set_postfix(loss=loss.item(), lr=current_lr, step=global_step)
                wandb.log({"loss": loss.item(), "learning_rate": current_lr}, step=global_step)

            del mse, loss, v, vtheta, latents, vtheta_h, latents_h, z_h, siglip_emb, siglip_vec, img_mask, txt_mask, cfg_mask, z, x_t, t, texp, img_mask_expanded

            # --- Logging & Checkpointing (Rank 0) ---
            # Increase frequency (e.g., every 1000 steps)
            if global_step % 1000 == 0:
                deepspeed.comm.barrier() # Ensure all processes sync before rank 0 proceeds
                if global_rank == 0:
                    with torch.no_grad():
                        log_weight_norms(model_engine, global_step)

                        # --- Sampling ---
                        # Increase frequency significantly (e.g., every 2500 or 5000 steps)
                        if global_step % 2500 == 0:
                            if ae is not None and noise is not None and ex_sig_emb is not None and ex_sig_vec is not None:
                                model_to_sample = model_engine.module 
                                model_to_sample.eval()
                                
                                ae = ae.to(device) 

                                grid = sample_images(model_to_sample, ae, noise, ex_sig_emb, ex_sig_vec)
                                torchvision.utils.save_image(grid, f"logs/sampled_images_step_{global_step}.png")

                                del grid
                                
                                model_to_sample.train()
                            else:
                                print("Skipping sampling image generation (missing AE or example data).")
                        # ---------------

            # Keep checkpointing frequency reasonable, or adjust as needed
            # Example: Checkpoint every 5000 steps
            if ((global_step % 5000) == 0) and global_step != 0:
                deepspeed.comm.barrier() # Sync before saving
                save_dir = os.path.join("models", f"step_{global_step}")
                model_engine.save_checkpoint(save_dir, client_state={'global_step': global_step}) 
                if global_rank == 0:
                    print(f"DeepSpeed Checkpoint saved to {save_dir}.")
            
            global_step += 1
            if global_step >= TRAIN_STEPS:
                break
        if global_step >= TRAIN_STEPS:
             break
        print(f"[Rank {global_rank}] Training complete.")

    deepspeed.comm.barrier()
    save_dir = os.path.join("models", "final_checkpoint")
    model_engine.save_checkpoint(save_dir, client_state={'global_step': global_step})
    if global_rank == 0:
        print(f"Final DeepSpeed Checkpoint saved to {save_dir}.")

if __name__ == "__main__":
    # Ensure the start method is set before main() is called if needed globally,
    # but putting it inside main() is often sufficient if it's the main entry point.
    main()