import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common_training_setup import run_specific_experiment

from datasets.ijepa_collate import JepaCollate
from datasets.ijepa_dataset import IJEPADatasetWrapper
from helpers.dataset_helpers import get_mnist_sets, get_stl10_sets
from ijepa_evaluation import IJepaEvaluator
from ijepa_training_setup import fit, make_target_encoder, build_ijepa_config, jepa_collate_fn
from models.vision_transformer.conv_embedding import ConvEmbedding
from models.vision_transformer.ijepa_classifier import IJEPAClassifier
from models.vision_transformer.predictor import Predictor
from models.vision_transformer.vision_transformer import VisionTransformer


def main_ijepa():
    patch_size = (8, 8)
    embedding_size = 384
    batch_size = 256
    nb_epochs = 1

    train_set, test_set, image_size = get_stl10_sets()
    # train_set, test_set, image_size = get_stl10_sets(train_split="unlabeled")

    jepa_config = build_ijepa_config(image_size, patch_size)
    jepa_train_set = IJEPADatasetWrapper(train_set)
    jepa_test_set = IJEPADatasetWrapper(test_set)
    collate = JepaCollate(jepa_config)

    train_loader = DataLoader(jepa_train_set, batch_size=batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate, drop_last=True, persistent_workers=True)
    test_loader = DataLoader(jepa_test_set, batch_size=batch_size, shuffle=False, num_workers=4,
                              collate_fn=collate, drop_last=True, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train IJepa
    embedding_layer = ConvEmbedding(image_size[0], patch_size=patch_size, embedding_size=embedding_size)
    context_encoder = VisionTransformer(embedding_layer=embedding_layer,
                                        image_size=image_size,
                                        nb_encoder_blocks=6,
                                        nb_heads=6,
                                        use_class_token=False)
    target_encoder = make_target_encoder(context_encoder)

    predictor = Predictor(embedding_dim=embedding_size, nb_layers=2, nb_heads=2)
    mask_token = nn.Parameter(torch.zeros(1, 1, embedding_size))

    optimizer = torch.optim.AdamW(list(context_encoder.parameters()) + list(predictor.parameters()) + [mask_token], lr=3e-4)

    eval_train_set, eval_test_set, image_size = get_stl10_sets()
    eval_train_loader = DataLoader(eval_train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
                                   persistent_workers=True)
    eval_test_loader = DataLoader(eval_test_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True,
                                  persistent_workers=True)
    ijepa_evaluator = IJepaEvaluator(eval_train_loader, eval_test_loader, device)

    _ = fit(train_loader, test_loader, context_encoder, target_encoder, predictor, mask_token, optimizer, device,
            nb_epochs=nb_epochs, eval_every=5, print_every=1, probe_evaluator=ijepa_evaluator)
    print("pre training... Done")
    # Freeze backbone and create a classifier using IJepa
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    target_encoder.eval()

    ijepa_evaluator.evaluate(target_encoder)


if __name__ == "__main__":
    print("Hello")
    main_ijepa()
    print("Done")
