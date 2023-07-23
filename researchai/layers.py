import numpy as np

from abc import ABC, abstractmethod
from typing import Literal

Linearity = Literal["Linear", "ReLU", "Tanh"]


class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def backward(self, grads: np.ndarray) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, non_linearity: Linearity = "Linear", bias: bool = True):
        """
        A fully connected layer

        Parameters
        ----------
        in_features: Input dimensions
            shape (num_batches, self.in_features)

        out_features: Output dimensions
            shape (num_batches, self.in_features)

        bias: weather to include bias or not.

        non_linearity: To calculate gain for kaiming initialization
        """
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.bias: bool = bias

        self.non_linearity: Linearity = non_linearity
        self._kaiming_init()

        # Values
        self.inputs: np.ndarray
        self.outputs: np.ndarray

        # Grads
        self.grads: np.ndarray
        self.weights_grad: np.ndarray
        self.biases_grad: np.ndarray
        self.inputs_grad: np.ndarray

    def _kaiming_init(self):
        if self.non_linearity == "ReLU":
            gain = np.sqrt(2)
        elif self.non_linearity == "Tanh":
            gain = 5 / 3
        elif self.non_linearity == "Linear":
            gain = 1
        else:
            raise ValueError("Non linearity is not valid.")

        # kaiming normal
        std = gain / np.sqrt(self.in_features)

        # Parameters
        self.weights = np.random.randn(
            self.in_features, self.out_features) * std
        self.weights_velocity: np.ndarray = np.zeros_like(self.weights)
        if self.bias:
            self.biases = np.zeros(self.out_features)
            self.biases_velocity: np.ndarray = np.zeros_like(self.biases)

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
        if self.bias:
            self.biases_grad = np.sum(self.grads, axis=0)

        return self.inputs_grad
