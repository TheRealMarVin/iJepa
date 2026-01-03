import torch.nn as nn

from models.vision_transformer.vision_transformer import VisionTransformer


class ViTClassifier(nn.Module):
    def __init__(self, embedding_layer,
                img_size,
                nb_output=10,
                nb_encoder_blocks=6,
                nb_heads=4):
        super(ViTClassifier, self).__init__()
        self.backbone = VisionTransformer(embedding_layer=embedding_layer,
                                          img_size=img_size,
                                          nb_encoder_blocks=nb_encoder_blocks,
                                          nb_heads=nb_heads,
                                          use_class_token=False)
        embedding_dim = self.backbone.embedding_layer.embedding_size
        self.classifier = nn.Linear(embedding_dim, nb_output)

    def forward(self, x):
        encoder = self.backbone(x)
        class_token = encoder.mean(dim=1)
        return self.classifier(class_token)