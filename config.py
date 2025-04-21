# DATASET_NAME = "SwayStar123/FFHQ_1024_DC-AE_f32"
# DATASET_NAME = "SwayStar123/pruned_preprocessed_commoncatalog-cc-by_DCAE"
# DATASET_NAME = "SwayStar123/preprocessed_commoncatalog-cc-by"
# DATASET_NAME = "SwayStar123/imagenet_288_dcae_fp8_captions"
# DATASET_NAME = "g-ronimo/IN1k96-augmented-latents_dc-ae-f32c32-sana-1.0"
# DATASET_NAME = "SwayStar123/imagenet1k_eqsdxlvae_latents"
# DATASET_NAME = "g-ronimo/imagenet1k_eqsdxlvae_latents"
DATASET_NAME = "SwayStar123/preprocessed_DCAE-f64_pd12m-full"
DS_DIR_BASE = "/mnt/data/dataset"
MODELS_DIR_BASE = "/mnt/data/models"
AE_SCALING_FACTOR = 0.4538 # f64c128-mix-1.0
# AE_SCALING_FACTOR = 0.13025 # sdxl-vae-fp16-fix
# AE_SCALING_FACTOR = 0.3189 # f32c32-in-1.0
# AE_SCALING_FACTOR = 0.41407 # f32-c32-sana-1.1
# AE_SCALING_FACTOR = 0.4552 # f32-c32-mix-1.0
# AE_SCALING_FACTOR = 0.12746273743957862 # eq-sdxl
# AE_SHIFT_FACTOR = 0.8640247167934477 # eq-sdxl
AE_SHIFT_FACTOR = 0.

BS = 6144
TRAIN_STEPS = 300_000
MASK_RATIO = 0.75 # Percent to mask
CFG_RATIO = 0.1 # Percent to drop
MAX_CAPTION_LEN = 32 # Token length to encode

LR = 0.0004

AE_HF_NAME = "dc-ae-f64c128-mix-1.0-diffusers"
# AE_HF_NAME = "madebyollin/sdxl-vae-fp16-fix"
# AE_HF_NAME = "dc-ae-f32c32-in-1.0-diffusers"
# AE_HF_NAME = "dc-ae-f32c32-sana-1.1-diffusers"
# AE_HF_NAME = "dc-ae-f32c32-mix-1.0-diffusers"
# AE_HF_NAME = "dc-ae-f32c32-sana-1.0-diffusers"
# AE_HF_NAME = "KBlueLeaf/EQ-SDXL-VAE"

if "f32" in AE_HF_NAME:
    AE_CHANNELS = 32
elif "f64" in AE_HF_NAME:
    AE_CHANNELS = 128
else:
    AE_CHANNELS = 4

SIGLIP_HF_NAME = "google/siglip-so400m-patch14-384"
SIGLIP_EMBED_DIM = 1152

SEED = 42

# Channel wise factors for dc-ae-f64c128-mix-1.0-diffusers. Calculated from 1 million random images from my dataset.
AE_SCALING_FACTORS = [0.4797, 0.4539, 0.4727, 0.4634, 0.5015, 0.4741, 0.4775, 0.4604, 0.4221,
        0.4487, 0.4763, 0.4641, 0.5044, 0.4072, 0.4680, 0.4822, 0.3699, 0.4858,
        0.4656, 0.5049, 0.3237, 0.5044, 0.4749, 0.3950, 0.5190, 0.4780, 0.4460,
        0.4368, 0.3406, 0.4426, 0.4854, 0.4775, 0.4473, 0.5000, 0.4531, 0.4922,
        0.2993, 0.4775, 0.4839, 0.5117, 0.3843, 0.4785, 0.4414, 0.4172, 0.4434,
        0.4473, 0.3965, 0.4443, 0.4829, 0.4507, 0.4441, 0.4839, 0.3262, 0.4724,
        0.4531, 0.5078, 0.4995, 0.4380, 0.4858, 0.4524, 0.4995, 0.4399, 0.5049,
        0.4775, 0.4570, 0.4790, 0.4834, 0.4482, 0.4915, 0.4187, 0.4685, 0.2346,
        0.4624, 0.1471, 0.4360, 0.4741, 0.4609, 0.4741, 0.5015, 0.3997, 0.4434,
        0.4834, 0.4763, 0.3398, 0.4558, 0.4844, 0.4795, 0.4937, 0.4785, 0.4890,
        0.4519, 0.4834, 0.4622, 0.4146, 0.4802, 0.4380, 0.4675, 0.4749, 0.4946,
        0.4727, 0.4849, 0.4709, 0.4736, 0.5029, 0.4834, 0.4368, 0.4514, 0.4417,
        0.4604, 0.4827, 0.4512, 0.5044, 0.4539, 0.4829, 0.4634, 0.4639, 0.4351,
        0.4919, 0.4651, 0.3926, 0.4197, 0.4775, 0.4895, 0.5000, 0.4519, 0.4727,
        0.4180, 0.5098]

DIT_G = dict(
    num_layers=40,
    num_heads=16,
    embed_dim=1408,
)
DIT_XL = dict(
    num_layers=28,
    num_heads=16,
    embed_dim=1152,
)
DIT_L = dict(
    num_layers=24,
    num_heads=16,
    embed_dim=1024,
)
DIT_B = dict(
    num_layers=12,
    num_heads=12,
    embed_dim=768,
)
DIT_S = dict(
    num_layers=12,
    num_heads=6,
    embed_dim=384,
)
