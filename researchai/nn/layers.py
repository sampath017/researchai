import numpy as np
from utils import Array


class Dense:
    def __init__(self, in_features: int, out_features: int):
        """A fully connected layer"""
        self.in_features = in_features
        self.out_features = out_features
        self.weights = 0.01 * \
            np.random.rand(self.in_features, self.out_features)
        self.biases = np.zeros(self.out_features)

        self.inputs: Array
        self.outputs: np.ndarray

    def forward(self, inputs: Array) -> np.ndarray:
        """
        Forward pass

        Parameters
        ----------
        inputs: shape (num_batches, self.in_features)

        Returns
        -------
        outputs: shape(num_batches, self.out_features)

        Examples
        --------
        >> inputs = np.random.rand(5, 10)

        >> dense = Dense(10, 1)
        >> outputs = dense.forward(inputs)
        """

        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                "The second dimension of inputs should match in_features")
        else:
            self.inputs = inputs

        self.outputs = np.dot(self.inputs, self.weights) + self.biases

        return self.outputs
