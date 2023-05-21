import numpy as np
from losses import CategoricalCrossEntropy


class TestCategoricalCrossEntropyLoss:
    def setup_method(self):
        self.loss_fn = CategoricalCrossEntropy()

    def teardown_method(self):
        del self.loss_fn

    def forward(self, y_predictions, y_true):
        self.output = self.loss_fn.calculate(y_predictions, y_true)

    def test_zero_loss(self):
        y_predictions = np.eye(4, 4)
        y_true = np.eye(4, 4)

        self.forward(y_predictions, y_true)

        assert self.output == 0.

    def test_numerical_underflow_stability(self):
        y_predictions = np.array([[0., 0.2, 0.8]])
        y_true = np.array([[1, 0, 0]])

        self.forward(y_predictions, y_true)

        assert not np.isinf(self.output)
