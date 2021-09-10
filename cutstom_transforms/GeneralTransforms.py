import torch
import numpy as np
from torch import Tensor
import torch.nn as nn


class WhiteNoise(nn.Module):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high
    
    def forward(self, img:Tensor)->Tensor:

        return apply_uniform_noise(img, self.low, self.high)


def apply_uniform_noise(image, low, high):
    """Apply uniform noise to an image, clip outside values to 0 and 1.
    parameters:
    - image: a numpy.ndarray 
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    """

    nrow = image.shape[1]
    ncol = image.shape[2]

    image[0,:,:] = image[0,:,:] + get_uniform_noise(low, high, nrow, ncol)
    image[1,:,:] = image[1,:,:] + get_uniform_noise(low, high, nrow, ncol)
    image[2,:,:] = image[2,:,:] + get_uniform_noise(low, high, nrow, ncol)

    #clip values
    image = torch.where(image < 0, torch.zeros_like(image), image)
    image = torch.where(image > 1, torch.ones_like(image), image)

    assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    return image

def get_uniform_noise(low, high, nrow, ncol):
    """Return uniform noise within [low, high) of size (nrow, ncol).
    parameters:
    - low: lower bound of noise within [low, high)
    - high: upper bound of noise within [low, high)
    - nrow: number of rows of desired noise
    - ncol: number of columns of desired noise
    """

    return torch.from_numpy(np.random.uniform(low=low, high=high,
                                size=(nrow, ncol)))
def is_in_bounds(mat, low, high):
    """Return wether all values in 'mat' fall between low and high.
    parameters:
    - mat: a numpy.ndarray 
    - low: lower bound (inclusive)
    - high: upper bound (inclusive)
    """

    return torch.all(torch.logical_and(mat >= 0, mat <= 1))