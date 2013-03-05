#! /usr/bin/env python

from ml import BackPropagation, NeuralNetwork, sigmoid, sigmoid_gradient
#from concurrent.futures import ProcessPoolExecutor
import numpy as np
import unittest

class TestSigmoid(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_sigmoid_gradient(self):
        self.assertEqual(sigmoid_gradient(0), 0.25)

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        datafile = np.load('data/data.npz')
        data = dict(datafile.items())

        self.weights = (data['theta1'], data['theta2'])
        self.numbers = data['numbers']
        self.labels = data['labels']
        self.nn = NeuralNetwork(self.weights)
        self.bp = BackPropagation(self.weights, self.numbers, self.labels)

    def test_predict(self):
        #with ProcessPoolExecutor() as executor:
        #    predictions = list(
        #        executor.map(self.nn.predict, self.numbers)
        #    )

        predictions = [self.nn.predict(n) for n in self.numbers]

        predictions = np.array(np.argmax(predictions, axis=1))
        success_count = (self.labels == predictions).sum()
        self.assertEqual(success_count, 4876)

    def test_backpropagation_penalty(self):
        penalty = self.bp._penalty(self.weights)
        self.assertEqual(penalty, 961.40693929604765)

    def test_backpropagation_cost_function(self):
        lambda_value = 1
        cost = self.bp._cost_function(self.weights, lambda_value)
        self.assertEqual(cost, 0.38376985909092359)

if __name__ == '__main__':
    unittest.main()
