import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attn=False):
        # x: (B, N, D)
        B, N, D = x.shape

        qkv = self.qkv(x)

        # reshape into heads
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores * (self.head_dim ** -0.5)
        attn = attn_scores.softmax(dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        if return_attn:
            return out, attn  # attn shape: (B, H, N, N)

        return out
