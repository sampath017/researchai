from abc import ABC, abstractmethod
import numpy as np

from nnfs.utils import one_hot


class Loss(ABC):
    @abstractmethod
    def _forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.float64:
        """
        Calculate the mean to get a single data loss from sample losses.
        """
        sample_losses = self._forward(y_pred, y_true)

        return np.mean(sample_losses)  # data loss


class CategoricalCrossEntropy(Loss):
    """
    Calculates the cross entropy loss
    """

    def __init__(self):
        y_pred: np.ndarray
        y_true: np.ndarray

    def _forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Parameters
        ----------
        y_pred: The prediction probabilities.
                shape (num_batches, *)

        y_true: The prediction probabilities.
                shape (same shape as y_pred) or (num_batches)

        Returns
        -------
        outputs: Array of sample losses
                shape (num_batches)

        Examples
        --------
        >>> inputs = np.random.rand(5, 10)
        >>> dense = Dense(10, 1)
        >>> dense.forward(inputs)
        array([[0.03152764],
            [0.02610983],
            [0.02270446],
            [0.03197972],
            [0.03055829]])
        """
        self.y_pred = y_pred
        self.y_true = one_hot(y_true)

        # clip values from both sides by 1e-7 to avoid overflow and to avoid bias toward 1
        y_pred_clipped = np.clip(self.y_pred, 1e-7, 1.0-1e-7)

        correct_confidences = np.sum(y_pred_clipped * self.y_true, axis=-1)
        neg_logs = -np.log(correct_confidences)

        return neg_logs  # sample losses

    def backward(self) -> np.ndarray:
        """
        Computer gradient for this layer values and parameters

        Returns
        -------
        inputs_grad: shape(num_batches, self.in_features)
        """
        # Gradients of values
        self.inputs_grad = -self.y_true / self.y_pred

        # Normialize with number of samples
        self.inputs_grad = self.inputs_grad / self.y_true.shape[0]

        return self.inputs_grad
