import math
import torch


def build_sinusoidal_positional_encoding(sequence_length, embedding_dim):
    positions = torch.arange(sequence_length).unsqueeze(1)

    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
    )

    positional_encoding = torch.zeros(sequence_length, embedding_dim)
    positional_encoding[:, 0::2] = torch.sin(positions * div_term)
    positional_encoding[:, 1::2] = torch.cos(positions * div_term)

    return positional_encoding
