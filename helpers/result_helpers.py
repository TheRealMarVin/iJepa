import matplotlib.pyplot as plt
import numpy as np
import torch

from helpers.dataset_helpers import get_mnist_sets


def display_gallery(images, title, nb_columns=3, nb_rows=3, captions=None):
    nb_images = len(images)
    per_page = nb_columns * nb_rows
    nb_pages = nb_images // per_page
    if nb_images % per_page != 0:
        nb_pages += 1

    for page in range(nb_pages):
        fig, axis = plt.subplots(nb_rows, nb_columns)
        fig.suptitle("page: {} - {}".format(page + 1, title))

        axis = np.atleast_2d(axis)

        for i in range(nb_rows):
            for j in range(nb_columns):
                idx = page * per_page + i * nb_columns + j

                ax = axis[i, j]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if idx >= nb_images:
                    ax.set_xlabel("")
                    continue

                img = images[idx]
                tmp_img = np.transpose(img, (1, 2, 0))
                ax.imshow(tmp_img.squeeze())

                if captions and idx < len(captions):
                    ax.set_xlabel(captions[idx])

        plt.tight_layout()
        plt.show()



def get_misclassified_samples(model, iterator, max_count, device, input_index=0, target_index=1):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for batch in iterator:
            x = batch[input_index].to(device)
            y_true = batch[target_index].to(device)

            y_pred = model(x)
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            pred_labels = y_pred.argmax(dim=1)
            mis_mask = pred_labels != y_true

            if mis_mask.any():
                mis_x = x[mis_mask].cpu()
                mis_pred = pred_labels[mis_mask].cpu()
                mis_true = y_true[mis_mask].cpu()

                for sample, p, t in zip(mis_x, mis_pred, mis_true):
                    misclassified.append((sample, int(p), int(t)))
                    if len(misclassified) >= max_count:
                        return misclassified

    return misclassified


def prepare_misclassified_for_gallery(misclassified_samples):
    preds = []
    captions = []

    for sample, pred, true in misclassified_samples:
        preds.append(sample)
        captions.append("P:{} T:{}".format(pred, true))

    return preds, captions
