import torch

def jepa_collate_fn(batch):
    imgs = torch.stack([img for img, _, _ in batch], dim=0)
    context_lists = [ci.tolist() for _, ci, _ in batch]
    min_context_size = min(len(ci) for ci in context_lists)
    context_indices_list = [torch.as_tensor(ci[:min_context_size], dtype=torch.long) for ci in context_lists]

    target_lists = [ti for _, _, ti in batch]
    norm_target_lists = []
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

        norm_target_lists.append(blocks)
        all_lengths.append(min(lengths))

    if not all_lengths:
        raise ValueError("No target indices found in batch (all blocks empty).")

    min_target_size = min(all_lengths)

    target_indices_list_list = [
        [
            torch.as_tensor(indices[:min_target_size], dtype=torch.long)
            for indices in sample_targets
        ]
        for sample_targets in norm_target_lists
    ]

    return imgs, context_indices_list, target_indices_list_list
