import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))


def random_weights(size, epsilon=0.12):
        return np.random.rand(*size)*2*epsilon-epsilon


class BackPropagation(object):
    def __init__(self, weights, training_set, labels, add_bias=True):
        self.training_set = training_set
        self.training_set_length = training_set.shape[0]
        self.labels = labels
        self.num_labels = weights[-1].shape[0]
        self.add_bias = add_bias

    def _penalty(self, weights):
        return np.sum([
            np.sum(np.sum(theta[:, 1:]**2, axis=1))
            for theta in weights
        ])

    def _cost_function(self, weights, lambda_value):
        # TODO : activation_function arg
        l = lambda_value
        p = self._penalty(weights)
        m = self.training_set.shape[0]
        Y = np.zeros((m, self.num_labels))
        X = self.training_set

        for i in range(m):
            Y[i][self.labels[i]] = 1

        if self.add_bias:
            ones = np.ones((X.shape[0], 1))
            input_layer = np.hstack((ones, X))
        else:
            input_layer = X

        activations = [input_layer]
        z = [None]

        # Process hidden layers
        for i in range(len(weights)):
            z.append(np.dot(activations[-1], weights[i].T))
            activations.append(sigmoid(z[-1]))

            # Don't add bias terms on the last layer
            if self.add_bias and i < len(weights)-1:
                ones = np.ones((activations[-1].shape[0], 1))
                activations[-1] = np.hstack((ones, activations[-1]))

        h = activations[-1] # Output layer
        r = (l*p)/(2*m)
        return np.sum(np.sum(-Y*np.log(h) - (1-Y)*np.log(1-h), axis=1))/m + r

    def train(self):
        self._cost_function()
        return None


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

        self.weights = training_method.train()
