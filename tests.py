#! /usr/bin/env python

#from concurrent.futures import ProcessPoolExecutor
import numpy as np
import unittest

from ml import NeuralNetwork
from helpers import sigmoid, sigmoid_gradient, random_weights,\
                    flatten_matrices, reshape_vector, debug_initialize_theta


datafile = np.load('data/data.npz')
data = dict(datafile.items())
NUMBERS = data['numbers']
LABELS = data['labels']
THETAS = (data['theta1'], data['theta2'])
THETAS_SHAPES = [w.shape for w in THETAS]
DEBUG_THETA = np.array([
    [0.0841471, -0.02794155, -0.09999902, -0.02879033],
    [0.09092974, 0.06569866, -0.05365729, -0.09613975],
    [0.014112, 0.09893582, 0.0420167, -0.07509872],
    [-0.07568025, 0.04121185, 0.09906074, 0.01498772],
    [-0.09589243, -0.05440211, 0.06502878, 0.09129453]
])


class TestHelpers(unittest.TestCase):
    def test_flatten_matrices(self):
        (vector, shapes) = flatten_matrices(THETAS)

        self.assertEqual(vector.shape[0], 10285)
        self.assertEqual(shapes, THETAS_SHAPES)

    def test_reshape_vector(self):
        vector, shapes = flatten_matrices(THETAS)
        matrices = list(reshape_vector(vector, shapes))
        for m, w in zip(matrices, THETAS):
            self.assertTrue((m == w).all())

    def test_debug_initialize_THETAS(self):
        w = debug_initialize_theta((5, 3))
        self.assertTrue(np.allclose(DEBUG_THETA, w))


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_sigmoid_gradient(self):
        self.assertEqual(sigmoid_gradient(0), 0.25)


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(THETAS, training_set=NUMBERS, labels=LABELS)

    def test_penalty(self):
        penalty = self.nn._penalty(THETAS)
        self.assertEqual(penalty, 961.40693929604765)

    def test_predict(self):
        # with ProcessPoolExecutor() as executor:
        #    predictions = list(
        #        executor.map(self.nn.predict, self.numbers)
        #    )

        predictions = [self.nn.predict(n) for n in NUMBERS]

        predictions = np.array(np.argmax(predictions, axis=1))
        success_count = (LABELS == predictions).sum()
        self.assertEqual(success_count, 4876)

    def test_non_regularized_cost_function(self):
        lambda_value = 0
        cost = self.nn._cost_function(self.nn.weights, lambda_value)
        self.assertEqual(cost, 0.28762916516131881)

    def test_regularized_cost_function(self):
        lambda_value = 1
        cost = self.nn._cost_function(self.nn.weights, lambda_value)
        self.assertEqual(cost, 0.38376985909092359)

#     def test_gradient(self):
#         input_layer_size = 3
#         hidden_layer_size = 5
#         num_labels = 3
#         m = 5

#         theta1 = debug_initialize_weights((hidden_layer_size, input_layer_size))
#         theta2 = debug_initialize_weights((num_labels, hidden_layer_size))

#         X = debug_initialize_weights((m, input_layer_size-1))
#         y = np.mod(np.arange(1, m+1), num_labels).T

#         weights = np.concatenate((
#             theta1.ravel(order='F'),
#             theta2.ravel(order='F')
#         ))

#         bp = BackPropagation((theta1, theta2), X, y, lambda_value=0)

#         cost = bp._cost_function(weights)
#         gradient = bp._gradient(weights)

    def test_train(self):
        theta1 = random_weights(THETAS[0].shape)
        theta2 = random_weights(THETAS[1].shape)
        thetas = (theta1, theta2)
        lambda_value = 2

        nn = NeuralNetwork(thetas, training_set=NUMBERS, labels=LABELS)
        weights = nn.train(lambda_value, maxiter=10, disp=False)[0]
        cost = nn._cost_function(weights, lambda_value)

        self.assertTrue(cost < 2)


if __name__ == '__main__':
    unittest.main()
