import torch.nn as nn


class ConvEmbedding(nn.Module):
    def __init__(self, nb_channels, patch_size, embedding_size):
        super(ConvEmbedding, self).__init__()

        self.patch_size = patch_size
        self.embedding_size = embedding_size

        self.conv_1 = nn.Conv2d(nb_channels, embedding_size, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        x = self.conv_1(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        # need to transpose because x.shape[1] is the embedding size (out channels) and x.shape[2] is the patch count
        x = x.transpose(1, 2)
        return x
