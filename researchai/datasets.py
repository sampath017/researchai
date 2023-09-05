import numpy as np

from typing import Tuple


def spiral(samples, classes):
    """
    Creates a spirals of each class

    Parameters
    ----------
    samples: number of samples to generate for each class
        type: int
    classes: number of classes
        type: int 

    Returns
    -------
    x: samples
        shape (samples*classes, 2)
    y: labels
        shape (samples*classes)

    Examples
    --------
    >>> x, y = spiral(samples=100, classes=3)

    >>> plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
    >>> plt.show()
    """
    x = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype=np.int64)

    for class_number in range(classes):
        ix = range(samples * class_number, samples * (class_number + 1))
        radius = np.linspace(0.0, 1, samples)
        theta = np.linspace(class_number*4, (class_number+1)
                            * 4, samples) + np.random.randn(samples)*0.09
        x[ix] = np.column_stack((radius *
                                np.cos(theta*2.5), radius*np.sin(theta*2.5)))
        y[ix] = class_number

    return x, y
