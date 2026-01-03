import time
from datetime import timedelta

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from train_eval.eval import evaluate


DEFAULT_NB_WORKERS = 4


def train(
    model,
    train_dataset,
    optimizer,
    criterion,
    scheduler,
    batch_size,
    n_epochs,
    shuffle,
    summary,
    save_file,
    early_stop=None,
    train_ratio=0.85,
    true_index=1,
    device=None,
    nb_workers=DEFAULT_NB_WORKERS,
    resplit_validation_each_epoch=False,
    fixed_split_seed=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    dataset_length = len(train_dataset)
    train_count = int(dataset_length * train_ratio)
    valid_count = dataset_length - train_count

    best_valid_loss = float("inf")

    if not resplit_validation_each_epoch:
        train_loader, valid_loader = _create_data_loaders(
            train_dataset,
            train_count,
            valid_count,
            batch_size,
            nb_workers,
            shuffle,
            fixed_split_seed
        )

    metrics = {"loss": criterion}

    for epoch in range(n_epochs):
        start_time = time.time()

        if resplit_validation_each_epoch:
            train_loader, valid_loader = _create_data_loaders(
                train_dataset,
                train_count,
                valid_count,
                batch_size,
                nb_workers,
                shuffle,
                fixed_split_seed,
                epoch_offset=epoch
            )

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            metrics,
            device=device,
            true_index=true_index
        )

        _, _, validation_metrics = evaluate(model, valid_loader, metrics, true_index=true_index)

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()

        end_time = time.time()
        delta_time = timedelta(seconds=(end_time - start_time))

        # Save best model
        if validation_metrics["loss"] < best_valid_loss:
            best_valid_loss = validation_metrics["loss"]
            if save_file is not None:
                torch.save(model.state_dict(), save_file)

        header_str = "Current Epoch: {} -> train_eval time: {}".format(epoch + 1, delta_time)
        train_str = metrics_to_string(train_metrics, "train")
        validation_str = metrics_to_string(validation_metrics, "val")
        print("{}\n\t{} - {}".format(header_str, train_str, validation_str))

        log_metrics_in_tensorboard(summary, train_metrics, epoch, "train")
        log_metrics_in_tensorboard(summary, validation_metrics, epoch, "val")
        summary.flush()

        if early_stop is not None:
            if early_stop.should_stop(validation_metrics):
                break

    return best_valid_loss


def _create_data_loaders(dataset, train_count, valid_count, batch_size, nb_workers, shuffle,
                         fixed_split_seed=None, epoch_offset=0):
    generator = None
    if fixed_split_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(fixed_split_seed + epoch_offset)

    train_subset, valid_subset = random_split(dataset,[train_count, valid_count], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=nb_workers, shuffle=shuffle)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, num_workers=nb_workers, shuffle=shuffle)

    return train_loader, valid_loader


def log_metrics_in_tensorboard(summary, metrics, epoch, prefix):
    for key, value in metrics.items():
        summary.add_scalar("{}/{}".format(prefix, key), value, epoch + 1)


def metrics_to_string(metrics, prefix):
    parts = []
    for key, value in metrics.items():
        parts.append("{} {}:{:.4f}".format(prefix, key, value))

    return " ".join(parts)


def train_epoch(model, iterator, optimizer, criterion, metrics_dict, device, true_index=1):
    metric_scores = {}
    for key in metrics_dict.keys():
        metric_scores[key] = 0.0

    model.train()

    for batch_index, batch in tqdm(enumerate(iterator), total=len(iterator), desc="train"):
        inputs = batch[0]
        targets = batch[true_index]

        if len(targets.shape) == 1:
            targets = targets.long()

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        if isinstance(predictions, tuple):
            predictions, _ = predictions

        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        for key, metric in metrics_dict.items():
            if key == "loss":
                metric_scores[key] += loss.item()
            else:
                metric_scores[key] += metric(predictions.detach(), targets.view(-1).detach()).item()

        del inputs
        del targets
        del predictions
        del loss

    for key, value in metric_scores.items():
        metric_scores[key] = value / len(iterator)

    return metric_scores


def log_metrics_in_tensorboard(summary, metrics, epoch, prefix):
    for key, value in metrics.items():
        summary.add_scalar("{}/{}".format(prefix, key), value, epoch + 1)


def metrics_to_string(metrics, prefix):
    parts = []
    for key, value in metrics.items():
        parts.append("{} {}:{:.4f}".format(prefix, key, value))

    return " ".join(parts)
