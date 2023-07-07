from abc import ABC, abstractmethod

import numpy as np

from researchai.nn.utils import Array, Float


class Loss(ABC):
    @abstractmethod
    def forward(self, y_pred: Array, y_true: np.ndarray) -> np.ndarray:
        pass

    def calculate(self, y_pred: Array, y_true: np.ndarray) -> np.float64:
        sample_losses = self.forward(y_pred, y_true)

        return np.mean(sample_losses)  # data loss


class CategoricalCrossEntropy(Loss):
    """
    Calculates the cross entropy loss
    """

    def forward(self, y_pred: Array, y_true: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Parameters
        ----------
        y_pred: The prediction probabilities.
                shape (num_batches, *)

        y_true: The prediction probabilities.
                shape (same shape as y_pred)

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
        # clip values from both sides by 1e-7 to avoid overflow and to avoid bias toward 1
        y_pred_clipped = np.clip(y_pred, 1e-7, 1.0-1e-7)

        # check one-hot encoded
        if y_true.ndim == 1:
            correct_confidences = y_pred_clipped[range(len(y_true)), y_true]
        elif y_true.ndim == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=-1
            )
        else:
            raise ValueError(
                "Incorrect shapes: y_true should have either 1 or 2 dimensions.")

        neg_logs = -np.log(correct_confidences)

        return neg_logs  # sample losses
