import numpy as np

from researchai.utils import Layer


class SGD:
    def __init__(self, learning_rate: float = 1.0):
        """Stochastic Gradient Descent"""
        self.learning_rate = learning_rate
        self.layer: Layer

    def step(self, layer: Layer) -> None:
        """
        Update the parameters of layer with single step

        Parameters
        ----------
        layer: The parameters of layer to be updated

        Examples
        --------
        >>> dense = Dense(10, 1)
        >>> optimizer = SGD(learning_rate=0.01)
        >>> inputs_grad = dense.backward()
        >>> optimizer.step(dense)
        """
        self.layer = layer

        self.layer.weights += -self.learning_rate*self.layer.weights_grad
        self.layer.biases += -self.learning_rate*self.layer.biases_grad
