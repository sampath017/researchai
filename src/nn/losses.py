from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)

        return np.mean(sample_losses)  # data loss


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # clip values to avoid overflow
        y_pred_clipped = np.clip(y_pred, 1e-7, 1)

        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(len(y_true)), y_true]
        elif y_true.ndim == 2:
            # one-hot encoded
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=-1
            )

        neg_logs = -np.log(correct_confidences)

        return neg_logs

    def backward(self, y_predictions, y_true):
        num_labels = y_true.shape[-1]
        num_samples = y_true.shape[0]

        if y_true.ndim == 1:
            y_true = np.eye(num_labels)[y_true]

        self.dinputs = -y_true / y_predictions
        self.dinputs /= num_samples
