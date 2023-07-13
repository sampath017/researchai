import pytest
import numpy as np
import torch
import torch.nn.functional as F

from nnfs.activations import ReLU, Softmax


class TestReLUForward:
    def test_zeros(self):
        """Check if number of negative inputs are same as the number of (outputs == 0) after passing thought ReLU"""
        inputs = np.random.randn(100, 100)

        relu = ReLU()
        outputs = relu.forward(inputs)

        np.testing.assert_equal((inputs <= 0).sum(), (outputs == 0).sum())

    @torch.no_grad()
    def test_against_torch(self):
        """Test the outputs of custom ReLU against torch ReLU."""
        inputs = np.random.randn(1000, 1000)

        relu = ReLU()

        np.testing.assert_array_equal(relu.forward(
            inputs), F.relu(torch.tensor(inputs)))


class TestSoftmaxForward:
    def setup_method(self):
        self.softmax = Softmax()

    def teardown_method(self):
        del self.softmax

    def forward(self, inputs):
        self.outputs = self.softmax.forward(inputs)

        return self.outputs

    @torch.no_grad()
    def test_against_torch(self):
        """Test the outputs of custom softmax against torch softmax."""
        n = 100_000_000
        inputs = np.random.rand(n)

        np.testing.assert_allclose(self.forward(
            inputs), F.softmax(torch.tensor(inputs), dim=-1))

    @pytest.mark.parametrize("i", range(10))
    def test_uniform_distribution(self, i):
        """probability of each point equals to 1/n."""
        n = 100_000_000
        inputs = np.random.rand(n)
        outputs = self.forward(inputs)
        np.testing.assert_allclose(outputs, 1 / n, atol=1e-8)

    @pytest.mark.parametrize("i", range(10))
    def test_sum_probabilities(self, i):
        """Sum of the probabilities on last axis equals 1."""
        inputs = np.random.rand(100_000)
        outputs = self.forward(inputs)

        np.testing.assert_almost_equal(outputs.sum(axis=-1), 1.0)

    @pytest.mark.parametrize("i", range(10))
    def test_numerical_stability(self, i):
        """Check for big number stability"""
        inputs = 1e24 * np.random.rand(100)
        outputs = self.forward(inputs)

        np.testing.assert_array_equal(np.isnan(outputs), False)
