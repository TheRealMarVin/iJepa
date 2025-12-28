import torch
import torch.nn as nn

from models.vision_transformer.positional_encoding import build_sinusoidal_positional_encoding
from models.vision_transformer.transformer_encoder_block import TransformerEncoderBlock


class VisionTransformer(nn.Module):
    def __init__(self, embedding_layer,
                 img_size,
                 nb_encoder_blocks,
                 nb_heads,
                 use_class_token=True):
        super(VisionTransformer, self).__init__()

        self.img_size = img_size
        self.embedding_layer = embedding_layer
        self.use_class_token = use_class_token

        embedding_dim = self.embedding_layer.embedding_size
        patch_height, patch_width = self.embedding_layer.patch_size

        height = self.img_size[1]
        width = self.img_size[2]

        assert height % patch_height == 0 and width % patch_width == 0, "Image size must be divisible by patch size"

        self.patch_count = (height // patch_height) * (width // patch_width)

        # Sequence length = patch tokens + optional class token
        max_sequence_length = self.patch_count + (1 if self.use_class_token else 0)

        positional_encoding = build_sinusoidal_positional_encoding(max_sequence_length, embedding_dim)
        self.register_buffer("positional_embeddings", positional_encoding.unsqueeze(0), persistent=False)

        if self.use_class_token:
            self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        encoder_list = []
        for _ in range(nb_encoder_blocks):
            encoder_list.append(TransformerEncoderBlock(embedding_dim=embedding_dim, nb_heads=nb_heads))
        self.encoder_block = nn.ModuleList(encoder_list)

    def forward(self, x, mask_indices=None):
        batch_size = x.shape[0]

        embedding = self.embedding_layer(x)

        if self.use_class_token:
            class_tokens = self.class_token.expand(batch_size, -1, -1)
            tokens = torch.cat([class_tokens, embedding], dim=1)
        else:
            tokens = embedding

        tokens = tokens + self.positional_embeddings[:, :tokens.size(1), :]

        if mask_indices is not None:
            if self.use_class_token:
                mask_indices = mask_indices + 1  # offset because CLS is at position 0

            tokens = torch.gather(tokens, dim=1, index=mask_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

        encoder = tokens
        for block in self.encoder_block:
            encoder = block(encoder)

        return encoder
