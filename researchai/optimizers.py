import numpy as np


class SGD:
    def __init__(
        self, lr=1.0,
        lr_decay=0.0,
        momentum=0.0
    ):
        """
        Stochastic Gradient Descent

        Parameters
        ----------
        lr: learning rate
        lr_decay: learning rate decay to apply at each step
        momentum: apply momentum update
        """
        self.initial_lr = lr
        self.current_lr = lr
        self.lr_decay = lr_decay

        if momentum < 0.0:
            raise ValueError("momentum should be 0.0 or greater")
        else:
            self.momentum = momentum
        self.iteration = 0

    def pre_step(self):
        """Apply learning rate decay, should be called before optimizer step"""
        if self.lr_decay:
            self.current_lr = self.initial_lr / \
                (1 + self.iteration*self.lr_decay)

    def step(self, layer):
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
            self.layer.weights_updates = self.layer.weights_velocity

            if self.layer.bias:
                self.layer.biases_velocity = self.momentum * \
                    self.layer.biases_velocity - self.current_lr*self.layer.biases_grad
                self.layer.biases_updates = self.layer.biases_velocity
        else:
            self.layer.weights_updates = -self.current_lr*self.layer.weights_grad
            if self.layer.bias:
                self.layer.biases_updates = -self.current_lr*self.layer.biases_grad

        self.layer.weights += self.layer.weights_updates
        if self.layer.bias:
            self.layer.biases += self.layer.biases_updates

    def post_step(self) -> None:
        """Should be called after optimizer step, updates the iteration variable"""
        self.iteration += 1


class AdaGrad:
    def __init__(
        self, lr=1.0,
        lr_decay=0.0,
        epsilon=1e-100
    ):
        """
        Adaptive Gradient

        Parameters
        ----------
        lr: learning rate
        lr_decay: learning rate decay to apply at each step
        epsilon: to prevent divison by zero
        """
        self.initial_lr = lr
        self.current_lr = lr
        self.lr_decay = lr_decay

        if epsilon < 1e-200 or epsilon > 1e-50:
            raise ValueError(
                "epsilon should be small enough to avoid division by zero")
        else:
            self.epsilon = epsilon
        self.layer
        self.iteration = 0

    def pre_step(self) -> None:
        """Apply learning rate decay, should be called before optimizer step"""
        if self.lr_decay:
            self.current_lr = self.initial_lr / \
                (1 + self.iteration*self.lr_decay)

    def step(self, layer) -> None:
        """
        Update the parameters of layer at each step

        Parameters
        ----------
        layer: The parameters of layer to be updated

        Examples
        --------
        >>> dense = Dense(10, 1)
        >>> optimizer = AdaGrad(lr=0.01)
        >>> inputs_grad = dense.backward()
        >>> optimizer.step(dense)
        """
        self.layer = layer

        self.layer.weights_cache += self.layer.weights_grad**2
        self.layer.weights += -self.current_lr*self.layer.weights_grad / \
            (np.sqrt(self.layer.weights_cache) + self.epsilon)

        if self.layer.biases:
            self.layer.biases_cache += self.layer.biases_grad**2
            self.layer.biases += -self.current_lr*self.layer.biases_grad / \
                (np.sqrt(self.layer.biases_cache) + self.epsilon)

    def post_step(self) -> None:
        """Should be called after optimizer step, updates the iteration variable"""
        self.iteration += 1
