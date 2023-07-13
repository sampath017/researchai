import numpy as np


def classification_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> np.float64:
    """
    Accuracy for classifcation tasks

    Parameters
    ----------
    y_pred: The prediction probabilities.
            shape (num_batches, *)

    y_true: The prediction probabilities.
            shape (same shape as y_pred) or (num_batches)

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

    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=-1)

    acc = np.mean(y_pred == y_true)

    return acc
