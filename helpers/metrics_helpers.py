import torch
from sklearn.metrics import accuracy_score


def arg_max_accuracy(y_predicted, y_true):
    y = torch.argmax(y_predicted, dim=1)
    y.detach()
    return accuracy_score(y, y_true)