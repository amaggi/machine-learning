import numpy as np
from scipy.optimize import fmin_cg

from helpers import add_bias, del_bias, sigmoid, sigmoid_gradient, \
                    random_weights, flatten_matrices, reshape_vector, \
                    debug_initialize_theta


class NeuralNetwork(object):
    def __init__(self, thetas, activation_function=sigmoid, add_bias=True,
                 training_set=None, labels=None):
        self.num_labels = thetas[-1].shape[0]
        self.weights, self.theta_shapes = flatten_matrices(thetas)
        self.add_bias = add_bias
        self.h = activation_function

        # "hasattr" to avoid this warning :
        # The truth value of an array with more than one element is ambiguous.
        if hasattr(training_set, 'T'):
            self.X = training_set
            self.m = self.X.shape[0]
            self.labels = labels

            self.Y = np.zeros((self.m, self.num_labels))
            for i in range(self.m):
                self.Y[i][self.labels[i]] = 1

    def _activations(self, thetas):
        if self.add_bias:
            input_layer = add_bias(self.X)
        else:
            input_layer = self.X

        # activations = [a1, a2, ...]
        activations = [input_layer]
        self.z = []

        # Process hidden layers
        for i in range(len(thetas)):
            self.z.append(np.dot(activations[-1], thetas[i].T))
            activations.append(sigmoid(self.z[-1]))

            # Don't add bias terms on the last layer
            if self.add_bias and i < len(thetas)-1:
                activations[-1] = add_bias(activations[-1])

        return activations

    def _cost_function(self, weights, lambda_value):
        thetas = list(reshape_vector(weights, self.theta_shapes))
        activations = self._activations(thetas)
        l = lambda_value

        p = self._penalty(thetas)
        h = activations[-1]  # Output layer
        r = (l*p)/(2*self.m)
        cost = np.sum(np.sum(-self.Y*np.log(h) - (1-self.Y)*np.log(1-h), axis=1))/self.m + r

        return cost

    def _gradient(self, weights, lambda_value):
        thetas = list(reshape_vector(weights, self.theta_shapes))
        activations = self._activations(thetas)
        l = lambda_value

        # sigmas = [sigma3, sigma2, ...]
        sigmas = [activations[-1]-self.Y]
        for layer in self.z[:-1]:
            if self.add_bias:
                layer = add_bias(layer)
                sigma = np.dot(sigmas[-1], thetas[1])*sigmoid_gradient(layer)
                sigmas.append(del_bias(sigma))

        # deltas = [delta1, delta2, ...]
        deltas = []
        for activation, sigma in zip(activations, sigmas[::-1]):
            deltas.append(np.dot(sigma.T, activation))

        # gradients = [theta1_grad, theta2_grad, ...]
        gradients = []
        for theta, delta in zip(thetas, deltas):
            theta = del_bias(theta)
            theta = add_bias(theta, values_function=np.zeros)
            gradient = delta/self.m + np.dot((l/self.m), theta)
            gradients.append(gradient.T.ravel())

        return np.concatenate(gradients)

    def _penalty(self, thetas):
        return np.sum([
            np.sum(np.sum(theta[:, 1:]**2, axis=1))
            for theta in thetas
        ])

    def predict(self, input_layer):
        output_layer = input_layer
        thetas = reshape_vector(self.weights, self.theta_shapes)
        for theta in thetas:
            if self.add_bias:
                output_layer = np.append(1, output_layer)

            output_layer = self.h(theta.dot(output_layer))

        return output_layer

    def train(self, lambda_value, **user_kwargs):
        args = [
            self._cost_function,
            self.weights
        ]

        kwargs = {
            'fprime': self._gradient,
            'args': [lambda_value]
        }

        kwargs.update(user_kwargs)

        # Force full_output
        kwargs['full_output'] = True

        # Return (weights, fopt, func_calls, grad_calls, warnflag) tuple
        return fmin_cg(*args, **kwargs)
