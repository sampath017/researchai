import pytest
import numpy as np

from researchai.layers import Dense
from researchai.activations import ReLU
from researchai.optimizers import SGD
from researchai.commons import Softmax_CategoricalCrossentropy
from researchai.metrics import classification_accuracy

from researchai.datasets import spiral


class TestNetwork:
    def setup_method(self):
        self.dense1 = Dense(2, 100)
        self.relu1 = ReLU()

        self.dense2 = Dense(100, 3)
        self.softmax_cross_entropy = Softmax_CategoricalCrossentropy()
        self.optimizer = SGD()

    def teardown_method(self):
        del self

    def forward(self, x, y_true):
        self.dense1.forward(x)
        self.relu1.forward(self.dense1.outputs)

        self.dense2.forward(self.relu1.outputs)
        self.softmax_cross_entropy.forward(self.dense2.outputs, y_true)

        self.outputs = self.softmax_cross_entropy.outputs
        self.loss = self.softmax_cross_entropy.loss

    def backward(self):
        self.softmax_cross_entropy.backward()
        self.dense2.backward(self.softmax_cross_entropy.inputs_grad)
        self.relu1.backward(self.dense2.inputs_grad)
        self.dense1.backward(self.relu1.inputs_grad)

    def step(self):
        self.optimizer.step(self.dense1)
        self.optimizer.step(self.dense2)

    @pytest.mark.parametrize("i", range(10))
    def test_accuracy(self, i):
        x, y = spiral(samples=1000, classes=3)
        self.forward(x, y)

        acc = classification_accuracy(self.outputs, y)

        assert np.isclose(acc, 0.33, atol=0.1)

    @pytest.mark.parametrize("i", range(10))
    def test_gradients_stability(self, i):
        x, y = spiral(samples=100, classes=3)

        for epoch in range(10_001):
            self.forward(x, y)
            if np.isnan(self.loss) or np.isinf(self.loss):
                pytest.fail(f"Loss became np.nan or np.inf at epoch: {epoch}")

            self.backward()
            if np.any(self.dense1.weights_grad > 1000.0) or np.any(self.dense2.weights_grad > 1000.0):
                pytest.fail(
                    f"Weights gradient has values greater than 1000.0 at epoch: {epoch}")

            self.step()
            if np.any(self.dense1.weights > 1000.0) or np.any(self.dense2.weights > 1000.0):
                pytest.fail(
                    f"Weights has values greater than 1000.0 at epoch: {epoch}")
