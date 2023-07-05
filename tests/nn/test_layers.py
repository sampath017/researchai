import numpy as np

from src.nn.layers import Dense


class TestDenseForward:
    def test_generic(self):
        """With multiple inputs and outputs"""
        inputs = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        dense = Dense(3, 2)
        dense.forward(inputs)

        assert dense.outputs.shape == (2, 2)

    def test_one_in_one_out(self):
        """one input and one output"""
        inputs = np.array([
            [1],
            [2],
            [3]
        ])

        dense = Dense(1, 1)
        dense.forward(inputs)

        assert dense.outputs.shape == (3, 1)

    def test_one_in_generic_out(self):
        """one input multiple outputs"""
        dense = Dense(1, 5)
        inputs = np.array([
            [1],
            [2],
            [3]
        ])
        dense.forward(inputs)

        assert dense.outputs.shape == (3, 5)

    def test_generic_in_one_out(self):
        """one input multiple outputs"""
        dense = Dense(10, 1)
        inputs = np.random.rand(5, 10)
        dense.forward(inputs)

        assert dense.outputs.shape == (5, 1)
