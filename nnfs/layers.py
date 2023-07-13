import numpy as np


class Dense:
    def __init__(self, in_features: int, out_features: int):
        """A fully connected layer"""
        self.in_features = in_features
        self.out_features = out_features

        # Parameters
        self.weights = 0.01 * \
            np.random.rand(self.in_features, self.out_features)
        self.biases = np.zeros(self.out_features)

        # Values
        self.inputs: np.ndarray
        self.outputs: np.ndarray

        # Grads
        self.grads: np.ndarray
        self.weights_grad: np.ndarray
        self.biases_grad: np.ndarray
        self.inputs_grad: np.ndarray

    def forward(self, inputs: np.ndarray) -> np.ndarray:
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
        >>> inputs = np.random.rand(5, 10)
        >>> dense = Dense(10, 1)
        >>> dense.forward(inputs)
        array([[0.03152764],
            [0.02610983],
            [0.02270446],
            [0.03197972],
            [0.03055829]])
        """
        self.inputs = inputs
        self.outputs = np.dot(self.inputs, self.weights) + self.biases

        return self.outputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """
        Computer gradient for this layer values and parameters

        Parameters
        ----------
        grads: incoming gradients during backpropagation using chain rule
            shape (num_batches, self.out_features)

        Returns
        -------
        inputs_grad: gradients computed with respective to input values
            shape(num_batches, self.in_features)
        """
        self.grads = grads

        # Gradients of values
        self.inputs_grad = np.dot(self.grads, self.weights.T)

        # Gradients of parameters
        self.weights_grad = np.dot(self.inputs.T, self.grads)
        self.biases_grad = np.sum(self.grads, axis=0)

        return self.inputs_grad
