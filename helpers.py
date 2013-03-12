import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))


def random_weights(size, epsilon=0.12):
    return np.random.rand(*size)*2*epsilon-epsilon


def debug_initialize_theta(size):
    w = np.zeros((size[0], size[1]+1))
    sins = np.sin(np.arange(1, w.size+1))
    return sins.reshape(w.shape, order='F')/10


def add_bias(z, values_function=np.ones):
    values = values_function((z.shape[0], 1))
    return np.hstack((values, z))


def del_bias(z):
    return np.delete(z, 0, 1)


def flatten_matrices(matrices):
    shapes = [m.shape for m in matrices]
    vector = np.concatenate([m.ravel(order='F') for m in matrices])
    return (vector, shapes)


def reshape_vector(vector, shapes):
    start = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        end = start + size
        yield np.reshape(vector[start:end], shape, order='F')
        start = end
