from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings

warnings.filterwarnings("ignore")

class IJepaEvaluator:
    def __init__(self, train_loader, test_loader, device):
        self.device = device

        self.train_loader = train_loader
        self.val_loader = test_loader
        self.probe = None

    def evaluate(self, model, display_only_accuracy=False):
        model = model.eval()
        model = model.to(self.device)

        self._train_probe(model, display_only_accuracy)

    def _make_inputs_from_images(self, model, dataloader):
        inputs, labels = [], []
        for images, targets in dataloader:
            labels.extend(targets.tolist())

            images = images.to(self.device)
            tokens = model(images)
            new_inputs = tokens.mean(dim=1).cpu().detach().tolist()

            inputs.extend(new_inputs)

        return inputs, labels

    def _train_probe(self, model, display_only_accuracy):
        train_embeddings, train_labels = self._make_inputs_from_images(model, self.train_loader)

        probe = LogisticRegression(max_iter=100)
        probe = CalibratedClassifierCV(probe)

        probe.fit(train_embeddings, train_labels)
        self.probe = probe

        eval_embeddings, eval_labels = self._make_inputs_from_images(model, self.val_loader)
        test_pred = probe.predict(eval_embeddings)

        acc = accuracy_score(eval_labels, test_pred)
        print("Probe Accuracy : ", acc)
        if not display_only_accuracy:
            print(classification_report(eval_labels, test_pred, digits=4))
            print(confusion_matrix(eval_labels, test_pred))
