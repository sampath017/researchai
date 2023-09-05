import numpy as np
from .utils import sparse


def classification_accuracy(y_pred, y_true):
    """
    Accuracy for classifcation tasks

    Parameters
    ----------
    y_pred: The prediction probabilities.
        shape: (num_batches, *)

    y_true: The prediction probabilities.
        shape: (same shape as y_pred) or (num_batches)

    Returns
    -------
    Accuracy of all samples

    Examples
    --------
    >>> y_pred = [
            [0.7, 0.2, 0.1],
    ...     [0.5, 0.1, 0.4],
    ...     [0.02, 0.9, 0.08]
    ... ]
    >>> y_true = np.array([0, 1, 1])
    >>> classification_accuracy(y_pred, y_true)
    0.6666666666666666
    """
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = sparse(y_true)

    acc = np.mean(y_pred == y_true)

    return acc
