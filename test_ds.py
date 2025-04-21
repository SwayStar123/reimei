from datasets import load_dataset, Features, Value, Sequence
from config import DS_DIR_BASE, DATASET_NAME, SIGLIP_HF_NAME, MODELS_DIR_BASE, AE_SCALING_FACTOR
from transformers import SiglipTokenizer, SiglipTextModel
import time
import datasets
import torch
from dataset.shapebatching_dataset import ShapeBatchingDataset, get_dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    datasets.config.HF_HUB_OFFLINE = 1
    # ds = load_dataset(f"{USERNAME}/{DATASET_NAME}", cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=1, split="train")
    # ds = ds.to_iterable_dataset(1000)

    # device="cuda"

    # siglip_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").to(device)
    # siglip_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")


    # ds = ShapeBatchingDataset(ds, 16, siglip_tokenizer, siglip_model, device, 1)

    # t = time.time()
    # for row in ds:
    #     print(f"Took: {time.time() - t}")
    #     x = row["ae_latent"]
    #     x = x * AE_SCALING_FACTOR
    #     print("std dev, mean", torch.std_mean(x))
    #     t = time.time()

    ds = get_dataset(512, 0, "cuda", torch.bfloat16, num_workers=8)

    t = time.time()
    for row in ds:
        print(f"Took: {time.time() - t}")
        t = time.time()
