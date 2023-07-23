from abc import ABC, abstractmethod
import numpy as np

from researchai.utils import one_hot


class Loss(ABC):
    @abstractmethod
    def _forward(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def calculate(self, logits: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Calculate the mean to get a single data loss from sample losses.
        """
        sample_losses = self._forward(logits, y)
        self.output = np.mean(sample_losses)

        return self.output  # data loss


class CategoricalCrossEntropy(Loss):
    """
    Calculates the cross entropy loss
    """

    def __init__(self):
        self.logits: np.ndarray
        self.logits_clipped: np.ndarray
        self.y: np.ndarray
        self.output: np.ndarray

    def _forward(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Parameters
        ----------
        logits: The prediction probabilities.
                shape (num_batches, *)

        y: The prediction probabilities.
                shape (same shape as logits) or (num_batches)

        Returns
        -------
        outputs: Array of sample losses
                shape (num_batches)
        """
        self.logits = logits
        self.y = one_hot(y, num_classes=self.logits.shape[-1])

        # clip values from both sides by a small number to avoid overflow and to avoid bias toward 1
        self.logits_clipped = np.clip(self.logits, 1e-100, 1.0-1e-100)

        probs = np.sum(self.logits_clipped * self.y, axis=-1)
        neg_logs = -np.log(probs)

        return neg_logs  # sample losses

    def backward(self) -> np.ndarray:
        """
        Computer gradient for this layer values and parameters

        Returns
        -------
        inputs_grad: shape(num_batches, self.in_features)
        """
        # Gradients of values
        self.inputs_grad = -self.y / self.logits  # GRADIENT EXPLOSION

        # Normialize with number of samples
        self.inputs_grad = self.inputs_grad / self.y.shape[0]

        return self.inputs_grad
