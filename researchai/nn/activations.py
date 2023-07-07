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
        inputs: shape (*)

        Returns
        -------
        outputs: shape (*)

        Examples
        --------
        >>> inputs = np.random.rand(5)     
        >>> relu = ReLU()
        >>> relu.forward(inputs)
        array([0.81978176, 0.09681321, 0.48868056, 0.75821085, 0.07669289])
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
         inputs: shape (num_batches, *)

         Returns
         -------
         outputs: shape (num_batches, *)

         Examples
         --------
         >>> inputs = np.random.rand(5, 10)
         >>> softmax = softmax(1, 10)
         >>> outputs = softmax.forward(inputs)
         >>> outputs.sum()
         1.0
         """
        self.inputs = inputs

        # overflow and underflow stability
        exponents = np.exp(
            self.inputs - np.max(self.inputs, axis=-1, keepdims=True))

        # Get the probabilities
        self.outputs = exponents / exponents.sum(axis=-1, keepdims=True)

        return self.outputs
