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
    position_embeddings = context_encoder.positional_embeddings.to(device)[0]

    losses = []
    for target_index in range(nb_targets):
        curr_target_indices = target_indices[target_index].to(device)
        target_size = curr_target_indices.shape[1]

        curr_mask_tokens = mask_token
        curr_mask_tokens = curr_mask_tokens.expand(batch_size, target_size, embedding_dim).clone()
        curr_mask_position_embeddings = torch.stack([position_embeddings[curr_target_indices[i], :] for i in range(batch_size)])
        curr_mask_tokens += curr_mask_position_embeddings

        predictor_input = torch.cat([context_tokens, curr_mask_tokens], dim=1)
        predictor_output = predictor(predictor_input)

        predictions = predictor_output[:, -target_size:, :]
        loss = F.mse_loss(predictions, torch.stack([teacher_tokens[i, target_indices[target_index][i]] for i in range(batch_size)]))
        losses.append(loss)

    total_loss = torch.mean(torch.stack(losses))
    return total_loss
