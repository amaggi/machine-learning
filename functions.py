import numpy as np


def sigmoid(array):
    return 1/(1+np.exp(-array))
