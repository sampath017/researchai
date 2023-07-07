import pytest
import numpy as np

from researchai.nn.layers import Dense
from researchai.nn.losses import CategoricalCrossEntropy
from researchai.nn.activations import ReLU, Softmax
from researchai.nn.metrics import accuracy

from researchai.nn.datasets import spiral


class TestNetwork:
    def setup_method(self):
        self.dense1 = Dense(2, 100)
        self.dense2 = Dense(100, 1000)
        self.dense3 = Dense(1000, 1000)
        self.dense4 = Dense(1000, 10)
        self.relu = ReLU()
        self.softmax = Softmax()
        self.loss_fn = CategoricalCrossEntropy()

    def teardown_method(self):
        del self

    def forward(self, x, y_true):
        self.dense1.forward(x)
        self.relu.forward(self.dense1.outputs)
        self.dense2.forward(self.relu.outputs)
        self.relu.forward(self.dense2.outputs)
        self.dense3.forward(self.relu.outputs)
        self.relu.forward(self.dense3.outputs)
        self.dense4.forward(self.relu.outputs)

        self.softmax.forward(self.dense4.outputs)
        self.output = self.loss_fn.calculate(self.softmax.outputs, y_true)

    @pytest.mark.parametrize("i", range(10))
    def test_accuracy(self, i):
        x, y = spiral(samples=1000, classes=10)
        self.forward(x, y)

        acc = accuracy(self.softmax.outputs, y)

        assert np.isclose(acc, 0.1, atol=0.03)
