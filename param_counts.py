import torch

# 1) Define your parameters.
from transformer.reimei import ReiMeiParameters, ReiMei
# from transformer.moedit import SparseMoeBlock

params = ReiMeiParameters(
    use_mmdit=True,
    use_ec=True,
    channels=32,
    patch_size=(1,1),
    embed_dim=768,
    num_layers=4,
    num_heads=(768 // 64),
    siglip_dim=1152,
    bert_dim=1024,
    num_experts=4,
    capacity_factor=2.0,
    shared_experts=2,
    dropout=0.1,
    token_mixer_layers=2,
    image_text_expert_ratio=4,
)

# 2) Build the ReiMei model.
model = ReiMei(params)

# A small helper to count *trainable* parameters.
def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# 3) Get total parameter count (all modules).
total_params = count_trainable_params(model)

print(f"Total trainable parameters in the entire model: {total_params:,}")

##############################################################################
# 4) Identify each SparseMoeBlock, compute "active" fraction, etc.
##############################################################################

sum_diff = 0.0
num_moe_blocks = 0

for name, submodule in model.named_modules():
    block_total = count_trainable_params(submodule)
    print(f"{name}: {block_total}")
    # if isinstance(submodule, SparseMoeBlock):
    #     # Count all params in this MoE block
    #     block_total = count_trainable_params(submodule)

    #     # f_c = capacity factor from the gate
    #     # E   = submodule.num_experts
    #     # The user-provided formula:
    #     #   active = (block_total / E) * f_c
    #     # This is a simplification! But we'll follow the user’s request.
    #     E   = getattr(submodule, "num_experts", 1)
    #     f_c = getattr(submodule, "f_c", 1.0)

    #     block_active = block_total / E * f_c
    #     diff = block_total - block_active
    #     sum_diff += diff
    #     num_moe_blocks += 1

        # print(f"\nSparseMoeBlock '{name}':")
        # print(f"  - total params:  {block_total:,.0f}")
        # print(f"  - active params: {block_active:,.1f}  (approx, using block_total / {E} * {f_c})")

##############################################################################
# 5) Compute the overall "average active" param count.
##############################################################################
avg_active_params = total_params - sum_diff

print(f"\nNumber of SparseMoeBlocks: {num_moe_blocks}")
print(f"Approximate 'Average Active' params = {avg_active_params:,.0f}")
