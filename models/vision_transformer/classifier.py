import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, embedding_dim, nb_output):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, nb_output)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, 0, :]

        x = self.fc(x)
        return x
