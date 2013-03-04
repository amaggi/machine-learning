from ml import NeuralNetwork
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import unittest

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        with np.load('data/data.npz') as datafile:
            data = dict(datafile.items())
        
        self.weights = (data['theta1'], data['theta2'])
        self.numbers = data['numbers']
        self.labels = data['labels']
        self.nn = NeuralNetwork(self.weights)

    def test_predict(self):
        with ProcessPoolExecutor() as executor:
            predictions = list(
                executor.map(self.nn.predict, self.numbers)
            )

        predictions = np.array(np.argmax(predictions, axis=1))
        success_count = (self.labels == predictions).sum()
        self.assertEqual(success_count, 4876)


if __name__ == '__main__':
    unittest.main()
