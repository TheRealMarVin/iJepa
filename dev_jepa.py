from helpers.patch_helpers import make_patch_grid, sample_block, sample_multiple_blocks, generate_context_and_targets

print(make_patch_grid((48,24), 8))
print(sample_block((5,3), 2, 2, 2, 2))

sample_multiple_blocks(nb_patches=(5,4), nb_blocks=5, min_block_height=2,
                       max_block_height=2, min_block_width=2, max_block_width=2,
                       max_tries_per_block=20)

generate_context_and_targets(
    nb_patches=(5,3),              # (nb_horizontal_patches, nb_vertical_patches)
    context_min_height=2,
    context_max_height=2,
    context_min_width=2,
    context_max_width=2,
    nb_targets=3,
    target_min_height=3,
    target_max_height=3,
    target_min_width=5,
    target_max_width=5,
    max_tries_per_block=20
)
