import torch
import torch.nn.functional as F


def jepa_loss(images, context_indices, target_indices, context_encoder, target_encoder,
              predictor, mask_token, device):
    images = images.to(device)
    context_indices = context_indices.to(device)

    batch_size = images.shape[0]
    embedding_dim = mask_token.shape[-1]
    nb_targets = len(target_indices)

    with torch.no_grad():
        teacher_tokens = target_encoder(images)

    context_tokens = context_encoder(images, mask_indices=context_indices)
    pos = context_encoder.positional_embeddings.to(device)[0]

    total_loss = 0.0
    for target_index in range(nb_targets):
        curr_target_indices = target_indices[target_index].to(device)
        nb_tokens = curr_target_indices.shape[1]

        teacher_target = torch.gather(teacher_tokens, dim=1, index=curr_target_indices.unsqueeze(-1).expand(-1, -1, teacher_tokens.shape[-1]))
        target_pos = pos[curr_target_indices]

        mask_tokens = mask_token.expand(batch_size, nb_tokens, embedding_dim) + target_pos

        pred_in = torch.cat([context_tokens, mask_tokens], dim=1)
        pred_out = predictor(pred_in)
        pred_target = pred_out[:, -nb_tokens:, :]

        total_loss += F.mse_loss(pred_target, teacher_target, reduction="sum")

    return total_loss / nb_targets
