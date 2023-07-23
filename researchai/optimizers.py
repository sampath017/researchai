import numpy as np

from researchai.layers import Layer


class SGD:
    def __init__(self, lr: float = 1.0, lr_decay: float = 0.0, momentum: float = 0.0):
        """
        Stochastic Gradient Descent

        Parameters
        ----------
        lr: learning rate
        lr_decay: learning rate decay to apply at each step
        momentum: apply momentum update
        """
        self.initial_lr: float = lr
        self.current_lr: float = lr
        self.lr_decay: float = lr_decay

        if momentum < 0.0:
            raise ValueError("momentum should be 0.0 or greater")
        else:
            self.momentum: float = momentum
        self.layer: Layer
        self.iteration: int = 0

    def pre_step(self) -> None:
        """Apply learning rate decay, should be called before optimizer step"""
        if self.lr_decay:
            self.current_lr = self.initial_lr / \
                (1 + self.iteration*self.lr_decay)

    def step(self, layer: Layer) -> None:
        """
        Update the parameters of layer at each step

        Parameters
        ----------
        layer: The parameters of layer to be updated

        Examples
        --------
        >>> dense = Dense(10, 1)
        >>> optimizer = SGD(lr=0.01)
        >>> inputs_grad = dense.backward()
        >>> optimizer.step(dense)
        """
        self.layer = layer

        if self.momentum > 0.0:
            self.layer.weights_velocity = self.momentum * \
                self.layer.weights_velocity - self.current_lr*self.layer.weights_grad
            weights_updates = self.layer.weights_velocity

            if self.layer.bias:
                self.layer.biases_velocity = self.momentum * \
                    self.layer.biases_velocity - self.current_lr*self.layer.biases_grad
                biases_updates = self.layer.biases_velocity
        else:
            weights_updates = -self.current_lr*self.layer.weights_grad
            if self.layer.bias:
                biases_updates = -self.current_lr*self.layer.biases_grad

        self.layer.weights += weights_updates
        if self.layer.bias:
            self.layer.biases += biases_updates

    def post_step(self) -> None:
        """Should be called after optimizer step, updates the iteration variable"""
        self.iteration += 1
