import numpy as np

from researchai.nn.utils import Array


class ReLU:
    def __init__(self):
        """ReLU function"""
        self.inputs: Array
        self.outputs: np.ndarray

    def forward(self, inputs: Array):
        """
        Forward pass

        Parameters
        ----------
        inputs: shape (Any)

        Returns
        -------
        outputs: shape (Any)

        Examples
        --------
        inputs = np.random.rand(5, 10)

        relu = ReLU(10, 1)
        outputs = relu.forward(inputs)

        print(outputs)
        """
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

        return self.outputs


class Softmax:
    def __init__(self):
        self.inputs: Array
        self.outputs: np.ndarray

    def forward(self, inputs: Array) -> np.ndarray:
        """
         Forward pass

         Parameters
         ----------
         inputs: shape (num_batches, Any)

         Returns
         -------
         outputs: shape (num_batches, Any)

         Examples
         --------
         >> inputs = np.random.rand(5, 10)

         >> softmax = softmax(1, 10)
         >> outputs = softmax.forward(inputs)

         >> outputs.sum()
         1.0
         """
        self.inputs = inputs

        # overflow and underflow stability
        exponents = np.exp(
            self.inputs - np.max(self.inputs, axis=-1, keepdims=True))

        # Get the probabilities
        self.outputs = exponents / exponents.sum(axis=-1, keepdims=True)

        return self.outputs
