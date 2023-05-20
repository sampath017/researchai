import pytest
import numpy as np

from src.nn.layers import Dense
from src.nn.losses import CategoricalCrossEntropyLoss
from src.nn.activations import ReLU, Softmax
from src.nn.metrics import accuracy

from nnfs.datasets import spiral_data


class TestNetwork:
    def setup_method(self):
        self.dense1 = Dense(2, 100)
        self.dense2 = Dense(100, 1000)
        self.dense3 = Dense(1000, 1000)
        self.dense4 = Dense(1000, 10)
        self.relu = ReLU()
        self.softmax = Softmax()
        self.loss_fn = CategoricalCrossEntropyLoss()

    def teardown_method(self):
        del self

    def forward(self, x, y_true):
        self.dense1.forward(x)
        self.relu.forward(self.dense1.output)
        self.dense2.forward(self.relu.output)
        self.relu.forward(self.dense2.output)
        self.dense3.forward(self.relu.output)
        self.relu.forward(self.dense3.output)
        self.dense4.forward(self.relu.output)

        self.softmax.forward(self.dense4.output)
        self.output = self.loss_fn.calculate(self.softmax.output, y_true)

    @pytest.mark.parametrize("i", range(10))
    def test_accuracy(self, i):
        x, y = spiral_data(samples=1000, classes=10)
        self.forward(x, y)

        acc = accuracy(self.softmax.output, y)

        assert np.isclose(acc, 0.1, atol=0.03)
