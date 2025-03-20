import torch
from tqdm import tqdm

from dataset.shapebatching_dataset import get_dataset

def calculate_channel_wise_scaling_factors(dataset, num_samples=1000000):
    """
    Calculate channel-wise scaling factors to normalize latent vectors to unit variance.
    Works with batched data from PyTorch DataLoader with latent shape [BS, C, H, W].
    
    Args:
        dataset: PyTorch DataLoader that returns batches with 'ae_latent' key
        num_samples: Number of samples to use for calculation
    
    Returns:
        torch.Tensor of scaling factors (1/std), one per channel
    """
    # Initialize counters and accumulators
    samples_processed = 0
    
    # Initialize variables for first batch to get dimensions
    first_batch = next(iter(dataset))
    # Get channel dimension from the first batch
    num_channels = first_batch["ae_latent"].shape[1]  # [BS, C, H, W]
    
    # Initialize tensor for running average of channel standard deviations
    device = first_batch["ae_latent"].device
    channel_stds_sum = torch.zeros(num_channels, device=device)
    
    # Progress bar
    pbar = tqdm(desc="Calculating scaling factors", total=num_samples)
    
    # Process batches
    dataset_iter = iter(dataset)
    batches_processed = 0
    
    while samples_processed < num_samples:
        try:
            # Get next batch (or first batch again)
            if batches_processed == 0:
                batch = first_batch
            else:
                batch = next(dataset_iter)
            batches_processed += 1
            
        except StopIteration:
            print(f"Reached end of dataset after {samples_processed} samples")
            break
            
        # Get latent vectors from batch
        latents = batch["ae_latent"]  # [BS, C, H, W]
        batch_size = latents.shape[0]
        
        # Calculate standard deviation for each channel across spatial dimensions and batch
        # Reshape to [BS*H*W, C] for std calculation
        latents_reshaped = latents.permute(0, 2, 3, 1).reshape(-1, num_channels)
        
        # Calculate std per channel 
        batch_channel_stds = latents_reshaped.std(dim=0)  # [C]
        
        # Update running average of standard deviations
        new_samples = min(batch_size, num_samples - samples_processed)
        weight = new_samples / (samples_processed + new_samples)
        
        if samples_processed == 0:
            # First batch
            channel_stds_sum = batch_channel_stds
        else:
            # Weighted average update
            channel_stds_sum = (1 - weight) * channel_stds_sum + weight * batch_channel_stds
        
        # Update counters
        samples_processed += new_samples
        pbar.update(new_samples)
        
        # Break if we've processed enough samples
        if samples_processed >= num_samples:
            break
    
    pbar.close()
    
    # Calculate scaling factors (1/std)
    epsilon = 1e-10  # Avoid division by zero
    channel_scaling_factors = 1.0 / (channel_stds_sum + epsilon)
    
    # Save scaling factors to a text file
    with open("scaling_factors.txt", "w") as f:
        for factor in channel_scaling_factors.cpu():
            f.write(f"{factor.item()}\n")
    
    print(f"\nCalculated scaling factors using {samples_processed} samples")
    print(f"Scaling factors saved to scaling_factors.txt")
    
    return channel_scaling_factors

# Example usage
if __name__ == "__main__":
    ds = get_dataset(512, 5, "cuda", torch.bfloat16, num_workers=4)
    scaling_factors = calculate_channel_wise_scaling_factors(ds)

    print(scaling_factors)