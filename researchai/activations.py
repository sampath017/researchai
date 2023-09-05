import numpy as np


class ReLU:
    """ReLU function"""

    def forward(self, inputs):
        """
        Forward pass

        Parameters
        ----------
        inputs: 
            shape: (*)

        Returns
        -------
        outputs:
            shape: (*)

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

    def backward(self, grads):
        """
        Computer gradient for this layer values and parameters.

        Parameters
        ----------
        grads: 
            shape (num_batches, *)

        Returns
        -------
        inputs_grad: gradients computed with respective to input values
            shape: grads.shape
            type: array_like
        """
        self.grads = grads

        # Gradients on values
        self.inputs_grad = self.grads.copy()
        self.inputs_grad[self.inputs <= 0] = 0

        return self.inputs_grad


class Tanh:
    def __init__(self):
        """Tanh function"""
        self.inputs
        self.outputs

        self.grads
        self.inputs_grad

    def forward(self, inputs):
        """
        Forward pass

        Parameters
        ----------
        inputs: the samples
            shape (*)

        Returns
        -------
        outputs: shape (*)

        Examples
        --------
        >>> inputs = np.random.rand(5)     
        >>> relu = Tanh()
        >>> tanh.forward(inputs)

        """
        self.inputs = inputs
        self.outputs = np.tanh(self.inputs)

        return self.outputs

    def backward(self, grads):
        """
        Computer gradient for this layer values and parameters.

        Parameters
        ----------
        grads: shape (num_batches, *)

        Returns
        -------
        shape (same as grads)
        """
        self.grads = grads

        # Gradients on values
        self.inputs_grad = 1 - self.outputs**2

        return self.inputs_grad


class Softmax:
    """Softmax activation function that outputs the probs."""

    def forward(self, inputs):
        """
        Forward pass

        Parameters
        ----------
        inputs: 
            shape: (num_batches, *)

        Returns
        -------
        outputs:
            shape: (num_batches, *)

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
            self.inputs - np.max(self.inputs, axis=-1, keepdims=True)
        )

        # Get the probabilities
        self.outputs = exponents / exponents.sum(axis=-1, keepdims=True)

        return self.outputs

    def backward(self, grads):
        """
        Computes gradient 

        Parameters
        ----------
        grads: incoming gradients during backpropagation using chain rule
            shape (num_batches, *)

        Returns
        -------
        inputs_grad: gradients computed with respective to input values
            shape (same as grads)
        """
        self.grads = grads

        self.inputs_grad = np.empty_like(self.grads)
        for index, (sample_output, sample_grads) in enumerate(zip(self.outputs, self.grads)):
            sample_output = np.reshape(sample_output, (-1, 1))

            jacobian_matrix = np.diagflat(sample_output) \
                - np.dot(sample_output, sample_output.T)

            # apply chain rule
            self.inputs_grad[index] = np.dot(jacobian_matrix, sample_grads)

        return self.inputs_grad
