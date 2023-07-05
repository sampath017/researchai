import numpy as np

from nn.activations import Softmax
from nn.losses import CategoricalCrossEntropy


class SoftmaxCategoricalCrossEntropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        return self.loss.calculate(self.activation.outputs, y_true)
