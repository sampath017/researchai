import numpy as np


def one_hot(array, num_classes=-1):
    """
    Convert sparse array to one-hot encoded

    Parameters
    ----------
    array: array to be encoded.
        shape: (num_batches, *)

    num_classes: number of classes to encode with
        type: int

    Returns
    -------
    output: encoded array
        shape: (num_batches, num_classes)
    """

    # check if starts from 0 and incremented by 1.

    if num_classes != -1:
        # check for sparse
        if array.ndim == 1:
            output = np.eye(num_classes)[array]
        elif array.ndim == 2:
            output = array
        else:
            raise ValueError("Number of dim must of either 1 or 2")
    else:
        raise ValueError("num_classes not defined")

    return output


def sparse(array):
    """
    Convert one-hot encoded array to sparse

    Parameters
    ----------
    array: array to be encoded.
        shape: (num_batches, num_classes)

    Returns
    -------
    output: encoded array
        shape: (num_batches, )
    """

    # check for sparse
    if array.ndim == 2:
        output = np.argmax(array, axis=-1)
    elif array.ndim == 1:
        output = array
    else:
        raise ValueError("Number of dim must of either 1 or 2")

    return output
