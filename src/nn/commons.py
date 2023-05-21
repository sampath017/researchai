import numpy as np

from nn.activations import Softmax
from nn.losses import CategoricalCrossEntropy


class SoftmaxCategoricalCrossEntropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        return self.loss.calculate(self.activation.output, y_true)

    def backward(self, y_predictions, y_true):
        num_samples = y_true.shape[0]

        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=-1)

        self.dinputs = y_predictions.copy()
        self.dinputs[range(num_samples), y_true] -= 1
        self.dinputs = self.dinputs / num_samples
