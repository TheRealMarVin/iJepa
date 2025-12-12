import torch.nn as nn

from models.vision_transformer.drop_path import DropPath
from models.vision_transformer.multi_head_self_attention import MultiHeadSelfAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        nb_heads,
        mlp_ratio = 4.0,
        attention_dropout = 0.0,
        proj_dropout = 0.0,
        mlp_dropout = 0.0,
        drop_path = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embedding_dim=embed_dim,
            nb_heads=nb_heads,
            attention_dropout=attention_dropout,
            proj_dropout=proj_dropout
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(mlp_dropout),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, return_attention=False):
        attn_out, attn = self.attn(self.norm1(x), return_attn=return_attention)
        x = x + self.drop_path(attn_out)

        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out)

        if return_attention:
            return x, attn
        return x
