import numpy as np
import torch
from tqdm import tqdm


def reshape_prediction_for_compatibility(raw_output):
    reshaped_res = []
    for x in range(raw_output[0].shape[0]):
        curr_out = []
        for y in raw_output:
            curr_out.append(y[x])

        reshaped_res.append(curr_out)

    return np.array(reshaped_res)


def evaluate(
    model,
    iterator,
    metrics_dict,
    true_index=1,
    device=None,
    return_all_predictions=True,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    metric_scores = {}
    for key in metrics_dict.keys():
        metric_scores[key] = 0.0

    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch_index, batch in tqdm(enumerate(iterator), total=len(iterator), desc="eval"):
            inputs = batch[0]
            targets = batch[true_index]

            if len(targets.shape) == 1:
                targets = targets.long()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs, _ = outputs

            for key, metric in metrics_dict.items():
                batch_metric = metric(outputs.detach(), targets.view(-1).detach())

                if isinstance(batch_metric, torch.Tensor):
                    metric_value = batch_metric.item()
                else:
                    metric_value = float(batch_metric)

                metric_scores[key] += metric_value

            if return_all_predictions:
                all_pred.extend(outputs.detach().cpu().numpy())
                all_true.extend(targets.detach().cpu().numpy())

            del inputs
            del targets
            del outputs

    for key, value in metric_scores.items():
        metric_scores[key] = value / len(iterator)

    return all_pred, all_true, metric_scores


def convert_string(tokenizer, device, field, text):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenized_text = tokenizer(text)
    indices = [field.vocab.stoi[x] for x in tokenized_text]

    res = torch.LongTensor([indices]).to(device)
    return res
