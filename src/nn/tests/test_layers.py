import torch
import numpy as np

from src.nn.layers import Dense


class TestDenseForward:
    def test_generic(self):
        """With multiple inputs and outputs"""
        dense = Dense(3, 2)
        inputs = np.array([[1, 2, 3], [4, 5, 6]])
        dense.forward(inputs)

        assert dense.output.shape == (2, 2)

    def test_one_in_one_out(self):
        """one input and one output"""
        dense = Dense(1, 1)
        inputs = np.array([[1], [2], [3]])
        dense.forward(inputs)

        assert dense.output.shape == (3, 1)

    def test_one_in_generic_out(self):
        """one input multiple outputs"""
        dense = Dense(1, 5)
        inputs = np.array([[1], [2], [3]])
        dense.forward(inputs)

        assert dense.output.shape == (3, 5)

    def test_generic_in_one_out(self):
        """one input multiple outputs"""
        dense = Dense(10, 1)
        inputs = np.random.rand(5, 10)
        dense.forward(inputs)

        assert dense.output.shape == (5, 1)


# class TestDenseBackward:
#     def test_backward(self):
#         # Create an instance of the Dense layer class
#         layer = Dense(2, 2)
#
#         # Set inputs, weights, and biases to known values
#         layer.inputs = np.array([[1, 2], [3, 4]])
#         layer.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
#         layer.biases = np.array([[0.5, 0.6]])
#
#         # Set dvalues to known values
#         dvalues = np.array([[0.7, 0.8], [0.9, 1.0]])
#
#         # Call the backward method
#         layer.backward(dvalues)
#
#         # Check that dweights, dbiases, and dinputs are correct
#         assert np.allclose(layer.dweights, np.array([[2.5, 3.1], [3.9, 4.9]]))
#         assert np.allclose(layer.dbiases, np.array([[1.6, 1.8]]))
#         assert np.allclose(layer.dinputs, np.array([[0.5, 0.7], [1.1, 1.5]]))
