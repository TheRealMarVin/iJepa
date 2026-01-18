import copy
import math

import torch
from tqdm import tqdm

from helpers.patch_helpers import compute_nb_patches
from ijepa_loss import jepa_loss

def build_ijepa_config(image_size=(3, 96, 96), patch_size=(8,8)):
    nb_patches = compute_nb_patches(image_size=image_size, patch_size=patch_size)

    config = {
        "nb_patches": nb_patches,

        "context": {
            "min_h": 2,
            "max_h": 10,
            "min_w": 2,
            "max_w": 10,
        },

        "target": {
            "nb_targets": 4,
            "min_h": 1,
            "max_h": 6,
            "min_w": 1,
            "max_w": 6,
            "max_tries": 100,
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


def cosine_ema(current_step, nb_steps_total, tau_base=0.996):
    return 1 - (1 - tau_base) * (math.cos(math.pi * current_step / nb_steps_total) + 1) / 2


def train_epoch(dataloader, context_encoder, target_encoder, predictor, mask_token, optimizer, device,
                current_step, nb_total_steps):
    context_encoder.train()
    predictor.train()

    running_loss = 0.0
    nb_steps = 0

    for images, context_indices, target_indices in dataloader:
        loss = jepa_loss(images, context_indices, target_indices, context_encoder, target_encoder,
                         predictor, mask_token, device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        current_step += 1
        momentum = cosine_ema(current_step, nb_total_steps, tau_base=0.996)
        ema_update(target_encoder, context_encoder, momentum)

        running_loss += float(loss.item())
        nb_steps += 1

    return running_loss / max(1, nb_steps)


@torch.no_grad()
def eval_epoch(dataloader, context_encoder, target_encoder, predictor, mask_token, device):
    context_encoder.eval()
    target_encoder.eval()
    predictor.eval()

    running_loss = 0.0
    nb_steps = 0

    for images, context_indices, target_indices in dataloader:
        images = images.to(device)

        loss = jepa_loss(images, context_indices, target_indices, context_encoder, target_encoder,
                         predictor, mask_token, device)

        running_loss += float(loss.item())
        nb_steps += 1

    return running_loss / max(1, nb_steps)


def fit(train_loader, val_loader, context_encoder, target_encoder, predictor, mask_token, optimizer, device, nb_epochs,
        eval_every=5, print_every=5, probe_evaluator=None):
    history = {"train_loss": [], "val_loss": []}

    context_encoder = context_encoder.to(device)
    target_encoder = target_encoder.to(device)
    predictor = predictor.to(device)
    mask_token = mask_token.to(device)

    nb_steps_from_epoch = len(train_loader)
    total_steps = nb_epochs * nb_steps_from_epoch

    for epoch in tqdm(range(nb_epochs)):
        train_loss = train_epoch(train_loader, context_encoder, target_encoder, predictor, mask_token, optimizer,
                                 device, (epoch * nb_steps_from_epoch),total_steps)
        val_loss = None
        if epoch == 1 or epoch % eval_every == 0:
            if val_loader is not None:
                val_loss = eval_epoch(val_loader, context_encoder, target_encoder, predictor, mask_token, device)
                history["val_loss"].append(val_loss)

            if probe_evaluator is not None:
                probe_evaluator.evaluate(target_encoder, display_only_accuracy=True)

        history["train_loss"].append(train_loss)

        if epoch % print_every == 0:
            if val_loss is None:
                print(f"epoch {epoch}/{nb_epochs + 1} | train_loss={train_loss:.6f}")
            else:
                print(f"epoch {epoch}/{nb_epochs + 1} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

    print(f"Final train_loss={train_loss:.6f}")

    return history
