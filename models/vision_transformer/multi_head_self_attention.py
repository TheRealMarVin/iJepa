import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, nb_heads, attention_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert embedding_dim % nb_heads == 0, "embed_dim must be divisible by nb_heads"
        self.embedding_dim = embedding_dim
        self.nb_heads = nb_heads
        self.head_dim = embedding_dim // nb_heads

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, return_attn=False):
        B, N, D = x.shape

        qkv = self.qkv(x)

        # reshape into heads
        qkv = qkv.reshape(B, N, 3, self.nb_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores * (self.head_dim ** -0.5)

        attention = attention_scores.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        out = torch.matmul(attention, v)

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)

        if return_attn:
            return out, attention

        return out, None
