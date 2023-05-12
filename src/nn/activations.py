import numpy as np


class ReLU:
    def forward(self, X):
        self.output = np.maximum(0, X)

        return self.output


class Softmax:
    def forward(self, X):
        exps = np.exp(X - np.max(X, axis=-1, keepdims=True)) # overflow stability
        probs = exps / exps.sum(axis=-1, keepdims=True)
        
        return probs
