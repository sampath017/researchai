import pytest
import numpy as np
from activations import ReLU, Softmax


class TestReLU:
    # FORWARD TEST
    def test_forward_zeros(self):
        relu = ReLU()
        inputs = np.random.randn(100, 100)
        output = relu.forward(inputs)

        assert (inputs <= 0).sum() == (output == 0).sum()


class TestSoftmax:
    def setup_method(self):
        self.softmax = Softmax()

    def teardown_method(self):
        del self.softmax

    def forward(self, inputs):
        self.output = self.softmax.forward(inputs)

    @pytest.mark.parametrize("i", range(10))
    def test_uniform_distribution(self, i):
        """probability of one point equals to 1/n."""
        n = 10_00_00_000
        inputs = np.random.rand(1, n)
        for batch in inputs:
            output = self.softmax.forward(batch)
            assert np.allclose(output, 1/n)

    @pytest.mark.parametrize("i", range(10))
    def test_sum_probabilities(self, i):
        """Sum of the probabilities on last axis equals 1."""
        inputs = np.random.rand(1000, 1000)
        self.forward(inputs)

        assert np.allclose(self.output.sum(axis=-1), 1.0)

    def test_numerical_overflow_stability(self):
        inputs = np.array([[1e13, 1e14], [1e20, 1e21]])
        expected_output = np.array([[0., 1.], [0., 1.]])
        self.forward(inputs)

        assert np.allclose(expected_output, self.output)
