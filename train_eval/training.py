import time
from datetime import timedelta

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from helpers.metrics_helpers import arg_max_accuracy
from train_eval.eval import evaluate


def train(model, train_dataset, optimizer,
          criterion, scheduler, batch_size,
          n_epochs, shuffle, summary, save_file,
          early_stop=None, train_ratio=0.85, true_index=1):
    tc = int(len(train_dataset) * train_ratio)
    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()
        x, y = torch.utils.data.random_split(train_dataset, [tc, len(train_dataset) - tc])
        train_iterator = torch.utils.data.DataLoader(x, batch_size=batch_size, num_workers=4, shuffle=shuffle)
        valid_iterator = torch.utils.data.DataLoader(y, batch_size=batch_size, num_workers=4, shuffle=shuffle)

        metrics = {"loss": criterion, "acc": arg_max_accuracy}
        train_metrics = train_epoch(model, train_iterator, optimizer, criterion, metrics, true_index=true_index)
        _, _, validation_metrics = evaluate(model, valid_iterator, metrics, true_index=true_index)
        if scheduler is not None:
            if type(scheduler) == lr_scheduler.ReduceLROnPlateau:
                scheduler.step(validation_metrics["loss"])
            else:
                scheduler.step()

        end_time = time.time()

        delta_time = timedelta(seconds=(end_time - start_time))

        if validation_metrics["loss"] < best_valid_loss:
            best_valid_loss = validation_metrics["loss"]
            if save_file is not None:
                torch.save(model, save_file)

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


def train_epoch(model, iterator, optimizer, criterion, metrics_dict, true_index = 1):
    metric_scores = {}
    for k, _ in metrics_dict.items():
        metric_scores[k] = 0

    model.train()

    for i, batch in tqdm(enumerate(iterator), total=len(iterator), desc="train"):
        src = batch[0]
        y_true = batch[true_index]

        if len(y_true.shape) == 1:
            y_true = y_true.type('torch.LongTensor')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            src = src.cuda()
            y_true = y_true.cuda()

        optimizer.zero_grad()

        y_pred = model(src)
        loss = criterion(y_pred, y_true)

        loss.backward()
        optimizer.step()

        for k, metric in metrics_dict.items():
            metric_scores[k] += metric(y_pred.detach(), y_true.view(-1).detach()).item()

        del src
        del loss
        del y_pred
        del y_true

    for k, v in metric_scores.items():
        metric_scores[k] = v / len(iterator)

    return metric_scores


def log_metrics_in_tensorboard(summary, metrics, epoch, prefix):
    for k, val in metrics.items():
        summary.add_scalar("{}/{}".format(prefix, k), val, epoch + 1)


def metrics_to_string(metrics, prefix):
    res = []
    for k, val in metrics.items():
        res.append("{} {}:{:.4f}".format(prefix, k, val))

    return " ".join(res)
