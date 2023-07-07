import numpy as np

from researchai.nn.utils import Array, Float


def accuracy(y_pred: Array, y_true: np.ndarray) -> np.float64:
    y_pred = np.argmax(y_pred, axis=-1)

    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=-1)

    acc = np.mean(y_pred == y_true)

    return acc
