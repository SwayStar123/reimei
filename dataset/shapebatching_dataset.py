import torch
from torch.utils.data import IterableDataset
from collections import defaultdict, deque
import numpy as np
from torch.utils.data import DataLoader
from config import DATASET_NAME, DS_DIR_BASE, MAX_CAPTION_LEN, MODELS_DIR_BASE, SIGLIP_HF_NAME
import random
from transformers import SiglipTextModel, SiglipTokenizer
from datasets import load_dataset
import time

# def custom_collate(batch):
#     captions = [item['caption'] for item in batch]
#     ae_latents = [item['ae_latent'] for item in batch]
#     ae_latent_shapes = [item['ae_latent_shape'] for item in batch]

#     return {
#         'caption': captions,
#         'ae_latent': ae_latents,
#         'ae_latent_shape': ae_latent_shapes
#     }

# def custom_collate(batch):
#     from walloc.walloc import pil_to_latent

#     captions = [item['caption'] for item in batch]
#     labels = [item['cls'] for item in batch]
#     ae_latents = [item['latent'] for item in batch]
#     ae_latents = [pil_to_latent([latent], N=36, n_bits=8, C=4)[:, :32].to(torch.int8).view(torch.float8_e4m3fn).to(torch.bfloat16) for latent in ae_latents]

#     ae_latent_shapes = [item.shape for item in ae_latents]

#     return {
#         'caption': captions,
#         'label': labels,
#         'ae_latent': ae_latents,
#         'ae_latent_shape': ae_latent_shapes
#     }

def custom_collate(batch):
    captions = [item['caption'][0] for item in batch]
    ae_latents = [item["ae_latent"] for item in batch]
    ae_latent_shapes = [item["ae_latent_shape"] for item in batch]

    return {
        'caption': captions,
        'ae_latent': ae_latents,
        'ae_latent_shape': ae_latent_shapes
    }


class ShapeBatchingDataset(IterableDataset):
    def __init__(self, hf_dataset, batch_size, siglip_tokenizer, siglip_model, device, num_workers, shuffle=True, seed=42, buffer_multiplier=4):
        self.dataset = hf_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = self.batch_size * buffer_multiplier
        self.siglip_tokenizer = siglip_tokenizer
        self.siglip_model = siglip_model
        self.target_device_for_encoding = device
        self.siglip_model_on_device = False

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_index = 0
            end_index = len(self.dataset)
            current_seed = self.seed
            worker_id = 0
        else:
            per_worker = int(np.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_index = worker_id * per_worker
            end_index = min(start_index + per_worker, len(self.dataset))
            current_seed = self.seed + worker_id

        worker_dataset = self.dataset.select(range(start_index, end_index))

        if self.shuffle:
            worker_dataset = worker_dataset.shuffle(seed=current_seed)

        if not self.siglip_model_on_device:
            try:
                self.siglip_model.to(self.target_device_for_encoding)
                self.siglip_model_on_device = True
            except Exception as e:
                print(f"Worker {worker_id}: FAILED to move Siglip model to {self.target_device_for_encoding}: {e}")
                raise

        dataset_iter = iter(worker_dataset)
        
        buffer = defaultdict(lambda: {'caption': deque(), 'ae_latent': deque()})
        exhausted = False

        while True:
            needed = self.batch_size
            current_buffer_size = sum(len(d['caption']) for d in buffer.values())

            while not exhausted and current_buffer_size < self.buffer_size:
                try:
                    sample = next(dataset_iter)
                    caption = sample['caption'][0]
                    ae_latent_data = sample["ae_latent"]
                    ae_latent_shape = sample["ae_latent_shape"]
                    shape_key = tuple(ae_latent_shape)

                    buffer[shape_key]['caption'].append(caption)
                    buffer[shape_key]['ae_latent'].append(ae_latent_data)
                    current_buffer_size += 1

                except StopIteration:
                    exhausted = True
                    break

            yielded_in_pass = False
            for shape_key in list(buffer.keys()):
                while len(buffer[shape_key]['caption']) >= self.batch_size:
                    samples_to_yield = {'caption': [], 'ae_latent': []}
                    for _ in range(self.batch_size):
                        samples_to_yield['caption'].append(buffer[shape_key]['caption'].popleft())
                        samples_to_yield['ae_latent'].append(buffer[shape_key]['ae_latent'].popleft())

                    if not buffer[shape_key]['caption']:
                         del buffer[shape_key]

                    batch_to_yield = self.prepare_batch(samples_to_yield, shape_key)
                    yield batch_to_yield
                    yielded_in_pass = True
                    current_buffer_size -= self.batch_size

            if exhausted and current_buffer_size == 0:
                break

            if exhausted and not yielded_in_pass and current_buffer_size > 0:
                 break

    def prepare_batch(self, samples, latent_shape):
        captions_list = list(samples["caption"])
        ae_latent_list = list(samples["ae_latent"])

        ae_latent = torch.tensor(ae_latent_list, dtype=torch.float16).to(torch.bfloat16).reshape(-1, *latent_shape).cpu()

        siglip_embedding, siglip_vec = self.encode_siglip(captions_list)
        siglip_embedding = siglip_embedding.cpu()
        siglip_vec = siglip_vec.cpu()

        batch = {
            'caption': captions_list,
            'ae_latent': ae_latent,
            'ae_latent_shape': latent_shape,
            'siglip_emb': siglip_embedding,
            'siglip_vec': siglip_vec,
        }
        return batch

    @torch.no_grad
    def encode_siglip(self, captions):
        s_tokens = self.siglip_tokenizer(captions, padding='longest', truncation=True, return_tensors="pt", max_length=MAX_CAPTION_LEN).to(self.target_device_for_encoding)
        siglip_outputs = self.siglip_model(**s_tokens, output_hidden_states=True)
        siglip_embedding_gpu = siglip_outputs.hidden_states[-1].to(dtype=torch.bfloat16)
        siglip_vec_gpu = siglip_outputs.pooler_output
        if siglip_vec_gpu is None:
             siglip_vec_gpu = siglip_embedding_gpu[:, 0]

        siglip_vec_gpu = siglip_vec_gpu.to(dtype=torch.bfloat16)

        return siglip_embedding_gpu, siglip_vec_gpu
    
def get_dataset(bs, seed, device, dtype, num_workers=16):
    ds = load_dataset(DATASET_NAME, cache_dir=f"{DS_DIR_BASE}/{DATASET_NAME}", num_proc=num_workers, split="train")

    siglip_model = SiglipTextModel.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip").eval()
    siglip_tokenizer = SiglipTokenizer.from_pretrained(SIGLIP_HF_NAME, cache_dir=f"{MODELS_DIR_BASE}/siglip")

    shape_batch_ds = ShapeBatchingDataset(ds, bs, siglip_tokenizer, siglip_model, device, num_workers=0, shuffle=True, seed=seed)
    
    return shape_batch_ds

# Note: The `num_workers` parameter in get_dataset now primarily controls Hugging Face dataset loading.
# The DataLoader inside ShapeBatchingDataset.__iter__ should use num_workers=0.
# The DataLoader created by DeepSpeed around ShapeBatchingDataset will handle multi-processing.
