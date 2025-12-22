import os
from configparser import ConfigParser
from os import path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from helpers.metrics_helpers import arg_max_accuracy
from helpers.result_helpers import display_gallery, prepare_misclassified_for_gallery, get_misclassified_samples
from train_eval.eval import evaluate
from train_eval.training import train, metrics_to_string


def load_training_config(train_config_file):
    config = ConfigParser()
    config.read(train_config_file)

    section = "default"

    nb_epochs = config.getint(section, "nb_epochs")
    batch_size = config.getint(section, "batch_size")
    learning_rate = config.getfloat(section, "learning_rate")

    if config.has_option(section, "train_ratio"):
        train_ratio = config.getfloat(section, "train_ratio")
    else:
        train_ratio = 0.85

    if config.has_option(section, "nb_workers"):
        nb_workers = config.getint(section, "nb_workers")
    else:
        nb_workers = 4

    if config.has_option(section, "weight_decay"):
        weight_decay = config.getfloat(section, "weight_decay")
    else:
        weight_decay = 0.0

    if config.has_option(section, "scheduler_type"):
        scheduler_type = config.get(section, "scheduler_type")
    else:
        scheduler_type = "cosine"  # or "plateau"

    if config.has_option(section, "save_dir"):
        save_dir = config.get(section, "save_dir")
    else:
        save_dir = "saved_models"

    if config.has_option(section, "nb_misclassified"):
        nb_misclassified = config.getint(section, "nb_misclassified")
    else:
        nb_misclassified = 18

    config_values = {
        "nb_epochs": nb_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_ratio": train_ratio,
        "nb_workers": nb_workers,
        "weight_decay": weight_decay,
        "scheduler_type": scheduler_type,
        "save_dir": save_dir,
        "nb_misclassified": nb_misclassified,
    }

    return config_values


def create_optimizer(model, learning_rate, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def create_scheduler(optimizer, scheduler_type, nb_epochs):
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs)
    elif scheduler_type == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, nb_epochs)

    return scheduler


def create_test_loader(test_dataset, batch_size, nb_workers):
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nb_workers,
    )
    return loader


def log_hyperparameters(summary, model_name, config_values):
    hparams = {
        "model_name": model_name,
        "learning rate": config_values["learning_rate"],
        "batch size": config_values["batch_size"],
        "max epochs": config_values["nb_epochs"],
        "train ratio": config_values["train_ratio"],
        "weight decay": config_values["weight_decay"],
        "scheduler type": config_values["scheduler_type"],
    }

    summary.add_hparams(hparams, {})


def get_model_save_path(save_dir, model_name):
    out_folder = os.path.join(save_dir, model_name)
    if not path.exists(out_folder):
        os.makedirs(out_folder)

    save_file = os.path.join(out_folder, "best.model")
    return save_file


def load_best_model_if_available(model, save_file, device):
    if path.exists(save_file):
        state_dict = torch.load(save_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("Loaded best model from {}".format(save_file))
    else:
        print("Best model file not found, using last epoch model.")

    model = model.to(device)
    model.eval()
    return model


def evaluate_on_test_set(model, test_loader, criterion, nb_misclassified):
    metrics = {"loss": criterion, "acc": arg_max_accuracy}

    y_pred, y_true, test_metrics = evaluate(model, test_loader, metrics)
    y_pred = np.array(y_pred).argmax(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bad_prediction_pairs = get_misclassified_samples(model, test_loader, nb_misclassified, device)
    images, captions = prepare_misclassified_for_gallery(bad_prediction_pairs)
    display_gallery(images, "Bad prediction", nb_columns=3, nb_rows=3, captions=captions)

    print(metrics_to_string(test_metrics, "test"))
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))


def run_specific_experiment(summary, model, datasets, train_config_file):
    train_dataset, test_dataset = datasets
    model_name = model.__class__.__name__

    config_values = load_training_config(train_config_file)

    nb_epochs = config_values["nb_epochs"]
    batch_size = config_values["batch_size"]
    learning_rate = config_values["learning_rate"]
    train_ratio = config_values["train_ratio"]
    nb_workers = config_values["nb_workers"]
    weight_decay = config_values["weight_decay"]
    scheduler_type = config_values["scheduler_type"]
    save_dir = config_values["save_dir"]
    nb_misclassified = config_values["nb_misclassified"]

    log_hyperparameters(summary, model_name, config_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    save_file = get_model_save_path(save_dir, model_name)

    test_loader = create_test_loader(test_dataset, batch_size, nb_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, learning_rate, weight_decay)
    scheduler = create_scheduler(optimizer, scheduler_type, nb_epochs)

    train(
        model,
        train_dataset=train_dataset,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        batch_size=batch_size,
        n_epochs=nb_epochs,
        shuffle=True,
        summary=summary,
        save_file=save_file,
        early_stop=None,
        train_ratio=train_ratio,
        true_index=1,
    )

    print("Finished Training")
    best_model = load_best_model_if_available(model, save_file, device)

    evaluate_on_test_set(best_model, test_loader, criterion, nb_misclassified)
