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

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)
