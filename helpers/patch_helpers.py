import random

import torch


def make_patch_grid(image_size, patch_size):
    """
    Args:
        image_size: (width, height) in pixels, or a single int for square images.
        patch_size: size of each square patch, in pixels.

    Returns:
        nb_vertical_patches: number of patch rows (height direction)
        nb_horizontal_patches: number of patch columns (width direction)
        coords: flat list of (row, col) tuples in row-major order
    """
    if not isinstance(image_size, (tuple, list)):
        image_size = (image_size, image_size)

    w, h = image_size
    if h % patch_size != 0 or w % patch_size != 0:
        raise Exception("image size not compatible with patch size")

    nb_vertical_patches = h // patch_size
    nb_horizontal_patches = w // patch_size

    coords = [(i, j) for j in range(nb_vertical_patches) for i in range(nb_horizontal_patches)]
    return nb_vertical_patches, nb_horizontal_patches, coords

def sample_block(nb_patches,
                 min_block_height,
                 max_block_height,
                 min_block_width,
                 max_block_width):
    """
    Sample a rectangular block in patch space.

    Args:
        nb_patches: (nb_horizontal_patches, nb_vertical_patches)
            where horizontal = width-wise, vertical = height-wise.
        min/max block sizes in patches.

    Returns:
        indices: list of linear patch indices (row-major)
        pos: (x, y) = top-left col, row in patch units
        size: (width, height) in patch units
    """
    nb_horizontal_patches, nb_vertical_patches = nb_patches

    if max_block_height < min_block_height:
        raise ValueError("max block height is smaller than min")

    if max_block_width < min_block_width:
        raise ValueError("max block width is smaller than min")

    height = random.randint(min_block_height, min(max_block_height, nb_vertical_patches))
    width = random.randint(min_block_width, min(max_block_width, nb_horizontal_patches))

    y = random.randint(0, nb_vertical_patches - height)
    x = random.randint(0, nb_horizontal_patches - width)

    pos = (x, y)
    size = (width, height)

    indices = [
        ( (y + h) * nb_horizontal_patches ) + (x + w)
        for h in range(height)
        for w in range(width)
    ]

    return indices, pos, size

def sample_multiple_blocks(
    nb_patches,
    num_blocks,
    min_block_height,
    max_block_height,
    min_block_width,
    max_block_width,
    max_tries_per_block=20,
    used_indices=None
):
    """
    Returns:
        blocks: list of dicts, each containing:
            {
                "indices": [...],   # list of linear patch indices
                "pos": (x, y),      # top-left patch coords
                "size": (w, h),     # width, height in patches
            }

        used_indices: set of all used indices (for inspection)
    """
    blocks = []
    if used_indices is None:
        used_indices = set()
    else:
        used_indices = set(used_indices)

    for i in range(num_blocks):
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

def generate_context_and_targets(
    nb_patches,              # (nb_horizontal_patches, nb_vertical_patches)
    min_context_height,
    max_context_height,
    min_context_width,
    max_context_width,
    num_targets,
    min_target_height,
    max_target_height,
    min_target_width,
    max_target_width,
    max_tries_per_block=20
):
    """
    Returns:
        context_indices: list[int]
        target_indices_list: list[list[int]]  # one list per target block
    """
    context_indices, pos, size = sample_block(nb_patches, min_context_height,
                                              max_context_height, min_context_width,
                                              max_context_width)
    target_blocks, _ = sample_multiple_blocks(nb_patches, num_targets, min_target_height,
                           max_target_height, min_target_width, max_target_width,
                           max_tries_per_block, set(context_indices))


    target_indices_list = [block["indices"] for block in target_blocks]

    return context_indices, target_indices_list

def build_index_mask_from_lists(context_indices_list, device=None):
    """
    Args:
        context_indices_list: list of list[int], length B.
            All inner lists must have the same length.
        device: optional torch.device

    Returns:
        masks: LongTensor of shape [B, N_context]
    """
    masks = torch.tensor(context_indices_list, dtype=torch.long, device=device)

    return masks

#TODO don't think this is needed
def gather_positional_embeddings(positional_embeddings, indices):
    """
    positional_embeddings: [N, D] or [1, N, D]
    indices: [B, T] (long)

    returns: [B, T, D]
    """
    if positional_embeddings.dim() == 3:
        pos = positional_embeddings[0]   # [N, D]
    else:
        pos = positional_embeddings      # [N, D]

    # pos[indices] uses advanced indexing -> [B, T, D]
    return pos[indices]

def compute_nb_patches(image_size, patch_size):
    image_height, image_width = image_size[1:]
    patch_height, patch_width = patch_size

    if image_height % patch_height != 0 or image_width % patch_width != 0:
        raise ValueError(f"image_size {image_size} not divisible by patch_size {patch_size}")

    nb_patches_h = image_height // patch_height
    nb_patches_w = image_width // patch_width
    return nb_patches_w, nb_patches_h

