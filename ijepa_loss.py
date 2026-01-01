import torch
import torch.nn.functional as F


def jepa_loss_one_target(imgs,
                         context_indices_list,
                         target_indices_list_list,
                         context_encoder,
                         target_encoder,
                         predictor,
                         mask_token,
                         target_block_index=0):
    device = imgs.device
    batch_size = imgs.shape[0]
    embedding_dim = mask_token.shape[-1]

    with torch.no_grad():
        teacher_tokens = target_encoder(imgs)

    context_mask = torch.stack(context_indices_list, dim=0).to(device)
    context_tokens = context_encoder(imgs, mask_indices=context_mask)

    target_indices_batch = torch.stack([target_indices_list_list[i][target_block_index].to(device) for i in range(batch_size)], dim=0)
    nb_tokens = target_indices_batch.shape[1]

    teacher_target = torch.gather(teacher_tokens, dim=1, index=target_indices_batch.unsqueeze(-1).expand(-1, -1, teacher_tokens.shape[-1]))

    pos = context_encoder.positional_embeddings.to(device)[0]
    target_pos = pos[target_indices_batch]

    mask_tokens = mask_token.expand(batch_size, nb_tokens, embedding_dim) + target_pos

    pred_in = torch.cat([context_tokens, mask_tokens], dim=1)
    pred_out = predictor(pred_in)
    pred_target = pred_out[:, -nb_tokens:, :]

    loss = F.mse_loss(pred_target, teacher_target)

    return loss
