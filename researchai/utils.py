import inspect
import numpy as np
from typing import Union, List

from researchai.layers import Dense

Array = Union[List[float], np.ndarray]
Layer = Dense
Float = Union[np.float64, float, int]


def one_hot(array: np.ndarray) -> np.ndarray:
    """
    Convert sparse array to one-hot encoded

    Parameters
    ----------
    array: array to be encoded.
            shape (num_batches, *)

    Returns
    -------
    output: encoded array
            shape (num_batches, num_classes)
    """

    # check if starts from 0 and incremented by 1.
    if np.max(array)+1 == len(np.unique(array)):
        # check for sparse
        if array.ndim == 1:
            output = np.eye(array.max() + 1)[array]
        elif array.ndim == 2:
            output = array
        else:
            raise ValueError("Number of dim must of either 1 or 2")
    else:
        raise ValueError(
            "Sparse array must start with 0 and to be incremented by 1, like unique classes has to (0, 1, 2) and not (2, 10, 12)"
        )

    return output


def sparse(array: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoded array to sparse

    Parameters
    ----------
    array: array to be encoded.
            shape (num_batches, num_classes)

    Returns
    -------
    output: encoded array
            shape (num_batches, )
    """

    # check for sparse
    if array.ndim == 2:
        output = np.argmax(array, axis=-1)
    elif array.ndim == 1:
        output = array
    else:
        raise ValueError("Number of dim must of either 1 or 2")

    return output
