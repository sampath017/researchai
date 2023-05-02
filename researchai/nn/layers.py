# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/00_layers.ipynb.

# %% auto 0
__all__ = ['Dense', 'spiral_data']

# %% ../../nbs/00_layers.ipynb 4
import numpy as np
import matplotlib.pyplot as plt

from fastcore.basics import patch

# %% ../../nbs/00_layers.ipynb 8
class Dense:
    "Fully connected layer"
    def __init__(
            self, 
            in_features: int, # The number of features.
            out_features: int # The number of neurons to have for the layer.
        ):
        
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.randn(out_features, in_features) # each row are the weigths for each neuron.
        self.biases = np.random.randn(out_features)
        
    def __str__(self):
        return f"Dense(in_features={self.in_features}, out_features={self.out_features})"
    
    __repr__ = __str__

# %% ../../nbs/00_layers.ipynb 12
@patch
def forward(
        self: Dense,
        X: np.ndarray # shape of (batch_size, in_features) if a single batch is passed it has to be of shape (1, in_features).
    ) -> np.ndarray: # the output of the layer
    "Forward input `X` through the layer"
    
    self.output = np.matmul(X, self.weights.T) + self.biases
    
    return self.output

# %% ../../nbs/00_layers.ipynb 17
def spiral_data(
        samples: int=100, # number of points per class
        classes: int=3, # number of classes
        noice_fact=None # add noice to the spiral angles.
    ) -> tuple[np.ndarray, np.ndarray]: # X and y arrays's
    "Create a 2D spiral for each class"
    
    N = samples 
    D = 2 # dimensionality
    C = classes  

    X = np.zeros((N*C,D)) # data matrix (each row = single example)
    y = np.zeros(N*C, dtype='uint8') # class labels

    for j in range(C):
        ix = range(N*j, N*(j+1))
        r = np.linspace(0.0, 1, N) # radius
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2 # theta
        
        if noice_fact: t *= noice_fact # scale the angle
        
        X[ix] = np.stack((r*np.sin(t), r*np.cos(t)), axis=-1)
        y[ix] = j
    
    return X, y