from layers import Dense
import numpy as np


class TestDense:
    def test_forward_generic(self):
        """With multiple inputs and outputs"""
        dense = Dense(3, 2)
        X = np.array([[1, 2, 3], [4, 5, 6]])
        dense.forward(X)

        assert dense.output.shape == (2, 2)

    def test_forward_one_in_one_out(self):
        """one input and one output"""
        dense = Dense(1, 1)
        X = np.array([[1], [2], [3]])
        dense.forward(X)

        assert dense.output.shape == (3, 1)

    def test_forward_one_in_generic_out(self):
        """one input multiple outputs"""
        dense = Dense(1, 5)
        X = np.array([[1], [2], [3]])
        dense.forward(X)

        assert dense.output.shape == (3, 5)

    def test_forward_generic_in_one_out(self):
        """one input multiple outputs"""
        dense = Dense(10, 1)
        X = np.random.rand(5, 10)
        dense.forward(X)

        assert dense.output.shape == (5, 1)
