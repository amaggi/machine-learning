#! /usr/bin/env python

#from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import fmin_cg
import numpy as np
import unittest

from ml import BackPropagation, NeuralNetwork
from helpers import sigmoid,\
                    sigmoid_gradient,\
                    random_weights,\
                    flatten_matrices,\
                    reshape_vector,\
                    debug_initialize_weights


datafile = np.load('data/data.npz')
data = dict(datafile.items())
TRAINING_SET = data['numbers']
LABELS = data['labels']
WEIGHTS = (data['theta1'], data['theta2'])
WEIGHTS_SHAPES = [w.shape for w in WEIGHTS]
DEBUG_WEIGHTS = np.array([
    [0.0841471, -0.02794155, -0.09999902, -0.02879033],
    [0.09092974, 0.06569866, -0.05365729, -0.09613975],
    [0.014112, 0.09893582, 0.0420167, -0.07509872],
    [-0.07568025, 0.04121185, 0.09906074, 0.01498772],
    [-0.09589243, -0.05440211, 0.06502878, 0.09129453]
])


class TestHelpers(unittest.TestCase):
    def test_flatten_matrices(self):
        (vector, shapes) = flatten_matrices(WEIGHTS)

        self.assertEqual(vector.shape[0], 10285)
        self.assertEqual(shapes, WEIGHTS_SHAPES)

    def test_reshape_vector(self):
        vector, shapes = flatten_matrices(WEIGHTS)
        matrices = list(reshape_vector(vector, shapes))
        for m, w in zip(matrices, WEIGHTS):
            self.assertTrue((m == w).all())

    def test_debug_initialize_weights(self):
        w = debug_initialize_weights((5, 3))
        self.assertTrue(np.allclose(DEBUG_WEIGHTS, w))


class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_sigmoid_gradient(self):
        self.assertEqual(sigmoid_gradient(0), 0.25)


class TestBackPropagation(unittest.TestCase):
    def setUp(self):
        self.bp = BackPropagation(WEIGHTS, TRAINING_SET, LABELS)

    def test_penalty(self):
        penalty = self.bp._penalty(WEIGHTS)
        self.assertEqual(penalty, 961.40693929604765)

    def test_non_regularized_cost_function(self):
        cost = self.bp._cost_function(self.bp.weights)
        self.assertEqual(cost, 0.28762916516131881)

    def test_regularized_cost_function(self):
        bp = BackPropagation(WEIGHTS, TRAINING_SET, LABELS, lambda_value=1)
        cost = bp._cost_function(self.bp.weights)
        self.assertEqual(cost, 0.38376985909092359)

    def test_gradient(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 5

        theta1 = debug_initialize_weights((hidden_layer_size, input_layer_size))
        theta2 = debug_initialize_weights((num_labels, hidden_layer_size))

        X = debug_initialize_weights((m, input_layer_size-1))
        y = np.mod(np.arange(1, m+1), num_labels).T

        weights = np.concatenate((
            theta1.ravel(order='F'),
            theta2.ravel(order='F')
        ))

        bp = BackPropagation((theta1, theta2), X, y, lambda_value=0)

        cost = bp._cost_function(weights)
        gradient = bp._gradient(weights)

    def test_train(self):
        theta1 = random_weights(WEIGHTS[0].shape)
        theta2 = random_weights(WEIGHTS[1].shape)
        weights = (theta1, theta2)

        new_bp = BackPropagation(weights, TRAINING_SET, LABELS, lambda_value=1)

        flatten_weights = flatten_matrices(weights)[0]
        #fmin_cg(new_bp._cost_function, flatten_weights, fprime=new_bp._gradient, maxiter=400)


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.weights = WEIGHTS
        self.numbers = TRAINING_SET
        self.labels = LABELS
        self.nn = NeuralNetwork(WEIGHTS)

    def test_predict(self):
        # with ProcessPoolExecutor() as executor:
        #    predictions = list(
        #        executor.map(self.nn.predict, self.numbers)
        #    )

        predictions = [self.nn.predict(n) for n in self.numbers]

        predictions = np.array(np.argmax(predictions, axis=1))
        success_count = (self.labels == predictions).sum()
        self.assertEqual(success_count, 4876)


if __name__ == '__main__':
    unittest.main()
