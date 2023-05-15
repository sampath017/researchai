import numpy as np


class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = 0.01 * np.random.rand(in_features, out_features)
        self.biases = np.zeros(out_features)

    def forward(self, X):
        self.output = np.dot(X, self.weights) + self.biases

        return self.output
