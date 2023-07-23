import numpy as np

from researchai.activations import Softmax
from researchai.losses import CategoricalCrossEntropy
from researchai.utils import sparse


class Softmax_CategoricalCrossentropy():
    """
    Softmax activation and categorical cross-entropy loss
    combined to make computation more efficient during backpropagation
    """

    def __init__(self):
        self.activation = Softmax()
        self.loss_fn = CategoricalCrossEntropy()

        self.inputs: np.ndarray
        self.y_true: np.ndarray
        self.outputs: np.ndarray

    def forward(self, inputs: np.ndarray, y_true: np.ndarray):
        """
        Forward pass

        Parameters
        ----------
        inputs: softmax inputs
        shape (num_batches, *)

        Returns
        -------
        outputs: cross-entropy loss
        shape (num_batches, *)

        Examples
        --------
        >>> inputs = np.random.rand(5, 10)
        >>> softmax_cross_entropy = Softmax_CategoricalCrossentropy()
        >>> loss = softmax_cross_entorpy.forward(inputs)
        >>> probs = softmax_cross_entorpy.outputs
        >>> probs.sum()
        1.0
        """
        self.inputs = inputs
        self.y_true = y_true

        self.activation.forward(self.inputs)
        self.outputs = self.activation.outputs
        self.loss = self.loss_fn.calculate(self.outputs, self.y_true)

        return self.loss

    def backward(self):
        """
        Computes gradient for this combination using chain rule

        Returns
        -------
        inputs_grad: gradients computed with respective to input values
            shape (same as self.inputs)
        """
        num_samples = self.outputs.shape[0]
        self.y_true = sparse(self.y_true)

        self.inputs_grad = self.outputs.copy()
        self.inputs_grad[range(num_samples), self.y_true] -= 1

        # Normialize gradients
        self.inputs_grad = self.inputs_grad / num_samples

        return self.inputs_grad
