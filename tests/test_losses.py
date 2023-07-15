import numpy as np
import torch
import pytest
import torch.nn.functional as F

from researchai.losses import CategoricalCrossEntropy


class TestCategoricalCrossEntropyLossForward:
    def setup_method(self):
        self.loss_fn = CategoricalCrossEntropy()

    def teardown_method(self):
        del self.loss_fn

    def forward(self, y_pred, y_true):
        self.output = self.loss_fn.calculate(y_pred, y_true)

        return self.output

    def test_zero_loss(self):
        y_pred = np.eye(4, 4)
        y_true = np.eye(4, 4)

        self.forward(y_pred, y_true)

        np.testing.assert_almost_equal(self.output, 0.0)

    def test_numerical_stability(self):
        y_predictions = np.array([[0., 0.2, 0.8]])
        y_true = np.array([[1, 0, 0]])

        self.forward(y_predictions, y_true)

        np.testing.assert_equal(np.isinf(self.output), False)

    @pytest.mark.parametrize("i", range(10))
    @torch.no_grad()
    def test_against_torch(self, i):
        """Test the outputs of custom cross_entropy against torch nll_loss."""
        inputs = torch.rand(10_000, 10_000)
        y_true = torch.arange(0, 10_000)

        y_pred = F.softmax(inputs, dim=-1)

        custom_loss = self.forward(y_pred.numpy(), y_true.numpy())
        torch_loss = F.nll_loss(torch.log(y_pred), y_true).numpy()

        np.testing.assert_allclose(custom_loss, torch_loss, atol=1e-6)
