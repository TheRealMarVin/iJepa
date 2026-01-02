import torch

def jepa_collate_fn(batch):
    images = torch.stack([curr_image for curr_image, _, _ in batch], dim=0)
    context_lists = [context_indices.tolist() for _, context_indices, _ in batch]
    context_min_size = min(len(context) for context in context_lists)
    context_indices = [torch.as_tensor(context[:context_min_size], dtype=torch.long) for context in context_lists]

    target_lists = [target_blocks for _, _, target_blocks in batch]

    target_blocks = []
    all_lengths = []
    for sample_targets in target_lists:
        blocks = []
        lengths = []
        for indices in sample_targets:
            indices_list = indices.tolist()
            blocks.append(indices_list)
            list_length = len(indices_list)
            if list_length > 0:
                lengths.append(list_length)

        target_blocks.append(blocks)
        all_lengths.append(min(lengths))

    if not all_lengths:
        raise ValueError("No target indices found in batch (all blocks empty).")

    min_target_size = min(all_lengths)

    target_indices = [
        [
            torch.as_tensor(indices[:min_target_size], dtype=torch.long)
            for indices in sample_targets
        ]
        for sample_targets in target_blocks
    ]

    return images, context_indices, target_indices
