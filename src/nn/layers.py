import numpy as np


class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        self.weights = np.random.rand(out_features, in_features)
        self.biases = np.random.rand(out_features)

    def forward(self, X):
        self.output = np.dot(X, self.weights.T) + self.biases

        return self.output
