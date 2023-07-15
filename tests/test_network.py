import pytest
import numpy as np

from researchai.layers import Dense
from researchai.losses import CategoricalCrossEntropy
from researchai.activations import ReLU, Softmax
from researchai.optimizers import SGD
from researchai.metrics import classification_accuracy

from researchai.datasets import spiral


class TestNetwork:
    def setup_method(self):
        self.dense1 = Dense(2, 100)
        self.relu1 = ReLU()

        self.dense2 = Dense(100, 3)
        self.softmax = Softmax()

        self.loss_fn = CategoricalCrossEntropy()
        self.optimizer = SGD()

    def teardown_method(self):
        del self

    def forward(self, x, y_true):
        self.dense1.forward(x)
        self.relu1.forward(self.dense1.outputs)

        self.dense2.forward(self.relu1.outputs)
        self.softmax.forward(self.dense2.outputs)

        self.output = self.loss_fn.calculate(self.softmax.outputs, y_true)

    def backward(self):
        self.loss_fn.backward()

        self.softmax.backward(self.loss_fn.inputs_grad)
        self.dense2.backward(self.softmax.inputs_grad)

        self.relu1.backward(self.dense2.inputs_grad)
        self.dense1.backward(self.relu1.inputs_grad)

    def step(self):
        self.optimizer.step(self.dense1)
        self.optimizer.step(self.dense2)

    @pytest.mark.parametrize("i", range(10))
    def test_accuracy(self, i):
        x, y = spiral(samples=1000, classes=10)
        self.forward(x, y)

        acc = classification_accuracy(self.softmax.outputs, y)

        assert np.isclose(acc, 0.1, atol=0.03)

    @pytest.mark.parametrize("i", range(10))
    def test_gradients_stability(self, i):
        x, y = spiral(samples=100, classes=3)

        with pytest.warns(RuntimeWarning):
            for epoch in range(10_001):
                self.forward(x, y)
                print(f"Epoch: {epoch}, Loss: {self.loss_fn.output:.3f}")

                self.backward()
                self.step()
