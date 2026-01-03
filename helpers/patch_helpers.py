import random


def sample_block(nb_patches, min_height, max_height, min_width, max_width):
    nb_horizontal_patches, nb_vertical_patches = nb_patches

    if max_height < min_height:
        raise ValueError("max block height is smaller than min")

    if max_width < min_width:
        raise ValueError("max block width is smaller than min")

    height = random.randint(min_height, min(max_height, nb_vertical_patches))
    width = random.randint(min_width, min(max_width, nb_horizontal_patches))

    y = random.randint(0, nb_vertical_patches - height)
    x = random.randint(0, nb_horizontal_patches - width)

    pos = (x, y)
    size = (width, height)

    indices = [((y + h) * nb_horizontal_patches ) + (x + w) for h in range(height) for w in range(width)]

    return indices, pos, size

def sample_multiple_blocks(nb_patches, nb_blocks, min_block_height, max_block_height, min_block_width,
                           max_block_width, max_tries_per_block=20, used_indices=None):
    blocks = []
    if used_indices is None:
        used_indices = set()
    else:
        used_indices = set(used_indices)

    for i in range(nb_blocks):
        found = False
        for j in range(max_tries_per_block):
            indices, pos, size = sample_block(nb_patches, min_block_height,
                                              max_block_height, min_block_width,
                                              max_block_width)
            if not used_indices.intersection(indices):
                used_indices.update(indices)
                blocks.append({"indices": indices, "pos": pos, "size": size})
                found = True
                break
        if not found:
            break

    return blocks, used_indices

def generate_context_and_targets(nb_patches, context_min_height, context_max_height, context_min_width,
                                 context_max_width, nb_targets, target_min_height, target_max_height, target_min_width,
                                 target_max_width, max_tries_per_block=20):
    context_indices, pos, size = sample_block(nb_patches, context_min_height,
                                              context_max_height, context_min_width,
                                              context_max_width)
    target_blocks, _ = sample_multiple_blocks(nb_patches, nb_targets, target_min_height,
                                              target_max_height, target_min_width, target_max_width,
                                              max_tries_per_block, set(context_indices))

    target_indices = [block["indices"] for block in target_blocks]

    return context_indices, target_indices


def compute_nb_patches(image_size, patch_size):
    image_height, image_width = image_size[1:]
    patch_height, patch_width = patch_size

    if image_height % patch_height != 0 or image_width % patch_width != 0:
        raise ValueError(f"image_size {image_size} not divisible by patch_size {patch_size}")

    nb_patches_h = image_height // patch_height
    nb_patches_w = image_width // patch_width
    return nb_patches_w, nb_patches_h

