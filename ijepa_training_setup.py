import copy
import torch
from tqdm import tqdm

from helpers.patch_helpers import compute_nb_patches
from ijepa_loss import jepa_loss_one_target

def build_ijepa_config(image_size=(3, 96, 96), patch_size=(8,8)):
    nb_patches = compute_nb_patches(image_size=image_size, patch_size=patch_size)

    config = {
        "nb_patches": nb_patches,

        "context": {
            "min_h": 2,
            "max_h": 3,
            "min_w": 2,
            "max_w": 3,
        },

        "target": {
            "nb_targets": 4,
            "min_h": 1,
            "max_h": 2,
            "min_w": 1,
            "max_w": 2,
            "max_tries": 50,
        }
    }
    return config


def make_target_encoder(context_encoder):
    target_encoder = copy.deepcopy(context_encoder)
    target_encoder.eval()
    for param in target_encoder.parameters():
        param.requires_grad_(False)
    return target_encoder


@torch.no_grad()
def ema_update(target_encoder, context_encoder, momentum):
    for target_param, context_param in zip(target_encoder.parameters(), context_encoder.parameters()):
        target_param.data.mul_(momentum).add_(context_param.data, alpha=1.0 - momentum)


def train_epoch(dataloader, context_encoder, target_encoder, predictor, mask_token, optimizer, device):
    context_encoder.train()
    predictor.train()

    running_loss = 0.0
    nb_steps = 0

    for images, context_indices, target_indices in dataloader:
        loss = jepa_loss_one_target(images,
                                    context_indices,
                                    target_indices,
                                    context_encoder,
                                    target_encoder,
                                    predictor,
                                    mask_token,
                                    device,
                                    target_block_index=0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        momentum = 0.99
        ema_update(target_encoder, context_encoder, momentum)

        running_loss += float(loss.item())
        nb_steps += 1

    return running_loss / max(1, nb_steps)


@torch.no_grad()
def eval_epoch(dataloader, context_encoder, target_encoder, predictor, mask_token, device):
    context_encoder.eval()
    predictor.eval()

    running_loss = 0.0
    nb_steps = 0

    for imgs, context_indices_list, target_indices_list_list in dataloader:
        imgs = imgs.to(device)

        loss = jepa_loss_one_target(imgs,
                                    context_indices_list,
                                    target_indices_list_list,
                                    context_encoder,
                                    target_encoder,
                                    predictor,
                                    mask_token,
                                    target_block_index=0)

        running_loss += float(loss.item())
        nb_steps += 1

    return running_loss / max(1, nb_steps)


def fit(train_loader, val_loader, context_encoder, target_encoder, predictor, mask_token, optimizer, device, nb_epochs, print_every=5):
    history = {"train_loss": [], "val_loss": []}

    for epoch in tqdm(range(1, nb_epochs + 1)):
        train_loss = train_epoch(train_loader, context_encoder, target_encoder, predictor, mask_token, optimizer, device)
        val_loss = None
        if val_loader is not None:
            val_loss = eval_epoch(val_loader, context_encoder, target_encoder, predictor, mask_token, device)
            history["val_loss"].append(val_loss)

        history["train_loss"].append(train_loss)

        if epoch % print_every == 0:
            if val_loss is None:
                print(f"epoch {epoch}/{nb_epochs} | train_loss={train_loss:.6f}")
            else:
                print(f"epoch {epoch}/{nb_epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    print(f"Final train_loss={train_loss:.6f}")

    return history


def jepa_collate_fn(batch):
    images = torch.stack([curr_image for curr_image, _, _ in batch], dim=0)
    context_lists = [context_indices.tolist() for _, context_indices, _ in batch]
    context_min_size = min(len(context) for context in context_lists)
    context_indices = torch.stack([torch.as_tensor(context[:context_min_size], dtype=torch.long) for context in context_lists],dim=0)

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

    target_indices = torch.stack([torch.stack([torch.as_tensor(indices[:min_target_size], dtype=torch.long) for indices in sample_targets], dim=0) for sample_targets in target_blocks], dim=0)

    return images, context_indices, target_indices

