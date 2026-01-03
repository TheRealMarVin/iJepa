import torch.nn as nn


class IJEPAClassifier(nn.Module):
    def __init__(self, backbone, embedding_dim, nb_output):
        super(IJEPAClassifier, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, nb_output)

    def forward(self, x):
        x = self.backbone(x)
        x = x.mean(dim=1)
        return self.classifier(x)
