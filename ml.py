import numpy as np
from scipy.optimize import fmin_cg

from helpers import add_bias,\
                    del_bias,\
                    sigmoid,\
                    sigmoid_gradient,\
                    random_weights,\
                    flatten_matrices,\
                    reshape_vector,\
                    debug_initialize_weights


class BackPropagation(object):
    def __init__(self, weights, training_set, labels, add_bias=True, lambda_value=0):
        self.X = training_set
        self.m = self.X.shape[0]
        self.labels = labels
        self.num_labels = weights[-1].shape[0]
        self.add_bias = add_bias
        self.weights, self.weight_shapes = flatten_matrices(weights)
        self.l = lambda_value

        self.Y = np.zeros((self.m, self.num_labels))
        for i in range(self.m):
            self.Y[i][self.labels[i]] = 1

    def _flatten_weights(self, weights):
        return np.concatenate([w.ravel() for w in weights])

    def _penalty(self, weights):
        return np.sum([
            np.sum(np.sum(theta[:, 1:]**2, axis=1))
            for theta in weights
        ])

    def _activations(self, weights):
        if self.add_bias:
            input_layer = add_bias(self.X)
        else:
            input_layer = self.X

        # activations = [a1, a2, ...]
        activations = [input_layer]
        self.z = []

        # Process hidden layers
        for i in range(len(weights)):
            self.z.append(np.dot(activations[-1], weights[i].T))
            activations.append(sigmoid(self.z[-1]))

            # Don't add bias terms on the last layer
            if self.add_bias and i < len(weights)-1:
                activations[-1] = add_bias(activations[-1])

        return activations

    def _cost_function(self, weights):
        weights = list(reshape_vector(weights, self.weight_shapes))
        activations = self._activations(weights)

        p = self._penalty(weights)
        h = activations[-1]  # Output layer
        r = (self.l*p)/(2*self.m)
        cost = np.sum(np.sum(-self.Y*np.log(h) - (1-self.Y)*np.log(1-h), axis=1))/self.m + r

        return cost

    def _gradient(self, weights):
        weights = list(reshape_vector(weights, self.weight_shapes))
        activations = self._activations(weights)

        # sigmas = [sigma3, sigma2, ...]
        sigmas = [activations[-1]-self.Y]
        for layer in self.z[:-1]:
            if self.add_bias:
                layer = add_bias(layer)
                sigma = np.dot(sigmas[-1], weights[1])*sigmoid_gradient(layer)
                sigmas.append(del_bias(sigma))

        # deltas = [delta1, delta2, ...]
        deltas = []
        for activation, sigma in zip(activations, sigmas[::-1]):
            deltas.append(np.dot(sigma.T, activation))

        # gradients = [theta1_grad, theta2_grad, ...]
        gradients = []
        for theta, delta in zip(weights, deltas):
            theta = del_bias(theta)
            theta = add_bias(theta, values_function=np.zeros)
            gradient = delta/self.m + np.dot((self.l/self.m), theta)
            gradients.append(gradient.T.ravel())

        return np.concatenate(gradients)

    def train(self):
        return weights


class NeuralNetwork(object):
    def __init__(self, weights, activation_function=sigmoid, add_bias=True):
        self.weights = weights
        self.num_labels = weights[-1].shape[0]
        self.h = activation_function
        self.add_bias = add_bias

    def predict(self, input_layer):
        output_layer = input_layer
        for weight in self.weights:
            if self.add_bias:
                output_layer = np.append(1, output_layer)

            output_layer = self.h(weight.dot(output_layer))

        return output_layer

    def train(self, training_set, labels, training_class=BackPropagation):
        training_method = training_class(
            self.weights,
            training_set,
            labels,
            add_bias=self.add_bias
        )

        self.weights = training_method.train(self.weights)
