import numpy as np


class Dense:
    def __init__(
        self,
        in_features,
        out_features,
        non_linearity="Linear",
        bias=True
    ):
        """
        A fully connected layer

        Parameters
        ----------
        in_features: Input dimensions
            type: int

        out_features: Output dimensions
            type: int

        bias: weather to include biases or not.
            type: boolean

        non_linearity: type of non-linearity used in the hidden units
            to calculate gain for kaiming initialization
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.non_linearity = non_linearity
        self._kaiming_init()

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

        # init parameters
        self.weights = np.random.randn(
            self.in_features, self.out_features) * std
        self.weights_updates = np.zeros_like(self.weights)
        self.weights_velocity = np.zeros_like(self.weights)
        self.weights_cache = np.zeros_like(self.weights)
        if self.bias:
            self.biases = np.zeros(self.out_features)
            self.biases_updates = np.zeros_like(self.biases)
            self.biases_velocity = np.zeros_like(self.biases)
            self.biases_cache = np.zeros_like(self.biases)

    def forward(self, inputs):
        """
        Forward pass

        Parameters
        ----------
        inputs: 
            shape: (num_batches, self.in_features)
            type: array_like

        Returns
        -------
        outputs: 
            shape(num_batches, self.out_features)
            type: array_like

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

    def backward(self, grads):
        """
        Computer gradient for this layer values and parameters

        Parameters
        ----------
        grads: incoming gradients during backpropagation using chain rule
            shape: (num_batches, self.out_features)
            type: array_like

        Returns
        -------
        inputs_grad: gradients computed with respective to input values
            shape: (num_batches, self.in_features)
            type: array_like
        """
        self.grads = grads

        # Gradients of values
        self.inputs_grad = np.dot(self.grads, self.weights.T)

        # Gradients of parameters
        self.weights_grad = np.dot(self.inputs.T, self.grads)
        if self.bias:
            self.biases_grad = np.sum(self.grads, axis=0)

        return self.inputs_grad
