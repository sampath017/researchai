import pytest
import numpy as np
from researchai.nn.activations import ReLU, Softmax


class TestReLU:
    # FORWARD TESTS
    def test_forward_zeros(self):
        """Check if number of negative inputs are same as the number of outputs == 0 after passing thought ReLU"""
        inputs = np.random.randn(100, 100)

        relu = ReLU()
        output = relu.forward(inputs)

        assert (inputs <= 0).sum() == (output == 0).sum()


class TestSoftmax:
    def setup_method(self):
        self.softmax = Softmax()

    def teardown_method(self):
        del self.softmax

    def forward(self, inputs):
        self.outputs = self.softmax.forward(inputs)

        return self.outputs

    @pytest.mark.parametrize("i", range(10))
    def test_uniform_distribution(self, i):
        """probability of each point equals to 1/n."""
        n = 10_00_00_000
        inputs = np.random.rand(n)
        outputs = self.forward(inputs)
        assert np.allclose(outputs, 1 / n)

    @pytest.mark.parametrize("i", range(10))
    def test_sum_probabilities(self, i):
        """Sum of the probabilities on last axis equals 1."""
        inputs = np.random.rand(1000)
        outputs = self.forward(inputs)

        assert np.allclose(outputs.sum(axis=-1), 1.0)

    @pytest.mark.parametrize("i", range(10))
    def test_numerical_stability(self, i):
        """Check for big number stability"""
        inputs = 1e24 * np.random.rand(100)
        outputs = self.forward(inputs)

        assert not np.isnan(outputs).any()
