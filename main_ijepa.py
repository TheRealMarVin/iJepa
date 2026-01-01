import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.ijepa_dataset import IJEPADatasetWrapper
from helpers.dataset_helpers import get_mnist_sets
from helpers.training_helpers import jepa_collate_fn
from ijepa_training_setup import fit, make_target_encoder, build_ijepa_config
from models.vision_transformer.conv_embedding import ConvEmbedding
from models.vision_transformer.predictor import Predictor
from models.vision_transformer.vision_transformer import VisionTransformer
from models.vision_transformer.vision_transformer_classifier import ViTClassifier


def main_ijepa():
    patch_size = (7, 7)
    embedding_size = 64
    batch_size = 64
    nb_epochs = 10
    train_set, test_set, image_size = get_mnist_sets()

    jepa_config = build_ijepa_config(image_size, patch_size)
    train_set = IJEPADatasetWrapper(train_set, jepa_config)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=jepa_collate_fn,
        drop_last=True
    )
    test_set = IJEPADatasetWrapper(test_set, jepa_config)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=jepa_collate_fn,
        drop_last=True
    )

    #summary = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_layer = ConvEmbedding(image_size[0], patch_size=patch_size, embedding_size=embedding_size)

    context_encoder = VisionTransformer(embedding_layer=embedding_layer,
                 img_size=image_size,
                 nb_encoder_blocks=6,
                 nb_heads=4,
                 use_class_token=False)
    target_encoder = make_target_encoder(context_encoder)

    predictor = Predictor(embedding_dim=embedding_size, nb_layers=2, nb_heads=2)
    mask_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()) + [mask_token], lr=1e-4, weight_decay=0.05)

    history = fit(train_loader, None, context_encoder, target_encoder, predictor, mask_token, optimizer, device, nb_epochs=nb_epochs, print_every=1)

    #summary.close()


if __name__ == "__main__":
    print("Hello")
    main_ijepa()
    print("Done")
