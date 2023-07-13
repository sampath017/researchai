import numpy as np
import torch
import torch.nn.functional as F

from nnfs.layers import Dense


class TestDenseForward:
    def test_single_batch(self):
        """Single batch with multiple inputs and outputs"""
        inputs = np.array([
            [1, 2, 3],
        ])

        dense = Dense(3, 2)
        dense.forward(inputs)

        np.testing.assert_equal(dense.outputs.shape, (1, 2))

    def test_one_in_one_out(self):
        """one input and one output"""
        inputs = np.array([
            [1],
            [2],
            [3]
        ])

        dense = Dense(1, 1)
        dense.forward(inputs)

        np.testing.assert_equal(dense.outputs.shape, (3, 1))

    def test_one_in_generic_out(self):
        """one input multiple outputs"""
        dense = Dense(1, 5)
        inputs = np.array([
            [1],
            [2],
            [3]
        ])
        dense.forward(inputs)

        np.testing.assert_equal(dense.outputs.shape, (3, 5))

    def test_generic_in_one_out(self):
        """one input multiple outputs"""
        dense = Dense(10, 1)
        inputs = np.random.rand(5, 10)
        dense.forward(inputs)

        np.testing.assert_equal(dense.outputs.shape, (5, 1))

    """
    TODO complete this test after implementing kaiming uniform for weight initialization
    `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015)
    """
    # @torch.no_grad()
    # def test_against_torch(self):
    #     """Test the outputs of custom ReLU against torch ReLU."""
    #     inputs = np.random.randn(1000, 20)
    #     torch_input = torch.tensor(inputs, dtype=torch.float32)

    #     dense = Dense(20, 20)
    #     linear = torch.nn.Linear(20, 20)

    #     np.testing.assert_allclose(dense.forward(
    #         inputs), linear(torch_input).detach().numpy())
