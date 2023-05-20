import numpy as np


class Dense:
    def __init__(self, in_features, out_features):
        self.weights = 0.01 * np.random.rand(in_features, out_features)
        self.biases = np.zeros(out_features)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, dvalues: np.ndarray):
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = dvalues.sum(axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
