import numpy as np
from activations import ReLU, Softmax


class TestReLU:
    # FORWARD TEST
    def test_forward_zeros(self):
        relu = ReLU()
        X = np.random.randn(100, 100)
        output = relu.forward(X)

        assert (X <= 0).sum() == (output == 0).sum()


class TestSoftmax:
    def setup_method(self, method):
        self.softmax = Softmax()

    def teardown_method(self, method):
        del self.softmax

    # FORWARD TEST
    def forward(self, X):
        self.output = self.softmax.forward(X)

    def test_uniform_distributation(self):
        """probablity of one point equals to 1/n."""
        n = 10_00_00_000
        X = np.random.rand(1, n)
        for batch in X:
            output = self.softmax.forward(batch)
            np.allclose(output, 1/n)

    def test_sum_probs(self):
        """Sum of the probs on last axis equals 1."""
        X = np.random.rand(1000, 1000)
        self.forward(X)

        assert np.allclose(self.output.sum(axis=-1), 1.0)

    def test_numerical_overflow_stablity(self):
        X = np.array([[1e13, 1e14], [1e20, 1e21]])
        expected_output = np.array([[0., 1.], [0., 1.]])
        self.forward(X)

        assert np.allclose(expected_output, self.output)
