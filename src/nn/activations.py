import numpy as np


class ReLU:
    def forward(self, X):
        self.inputs = X
        self.output = np.maximum(0, X)

        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.inputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, X):
        self.inputs = X
        # overflow stability
        exps = np.exp(X - np.max(X, axis=-1, keepdims=True))
        self.output = exps / exps.sum(axis=-1, keepdims=True)

        return self.output
