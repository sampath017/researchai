import numpy as np

from typing import Tuple


def spiral(samples: int, classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a spirals of each class

    Parameters
    ----------
    samples: The number of samples to generate for each class 

    Returns
    -------
    X: the samples
        shape (samples*classes, 2)
    y: the labels
        shape (samples*classes,)

    Examples
    --------
    >>> X, y = spiral(samples=100, classes=3)

    >>> plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    >>> plt.show()
    """
    X: np.ndarray = np.zeros((samples * classes, 2))
    y: np.ndarray = np.zeros(samples * classes, dtype=np.int64)

    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4,
                        samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y
