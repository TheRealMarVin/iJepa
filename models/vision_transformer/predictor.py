import torch.nn as nn

from models.vision_transformer.transformer_encoder_block import TransformerEncoderBlock


class Predictor(nn.Module):
    def __init__(self, embedding_dim, nb_layers, nb_heads, mlp_ratio=4.0):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embedding_dim=embedding_dim, nb_heads=nb_heads, mlp_ratio=mlp_ratio)
            for _ in range(nb_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
