import torch
import torch.nn.functional as F


def jepa_loss_one_target(images,
                         context_indices,
                         target_indices,
                         context_encoder,
                         target_encoder,
                         predictor,
                         mask_token,
                         device,
                         target_block_index=0):
    images = images.to(device)
    context_indices = context_indices.to(device)

    batch_size = images.shape[0]
    embedding_dim = mask_token.shape[-1]

    with torch.no_grad():
        teacher_tokens = target_encoder(images)

    context_tokens = context_encoder(images, mask_indices=context_indices)

    target_indices = target_indices[:, target_block_index, :].to(device)
    nb_tokens = target_indices.shape[1]

    teacher_target = torch.gather(teacher_tokens, dim=1, index=target_indices.unsqueeze(-1).expand(-1, -1, teacher_tokens.shape[-1]))

    pos = context_encoder.positional_embeddings.to(device)[0]
    target_pos = pos[target_indices]

    mask_tokens = mask_token.expand(batch_size, nb_tokens, embedding_dim) + target_pos

    pred_in = torch.cat([context_tokens, mask_tokens], dim=1)
    pred_out = predictor(pred_in)
    pred_target = pred_out[:, -nb_tokens:, :]

    loss = F.mse_loss(pred_target, teacher_target)

    return loss
