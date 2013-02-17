#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ml import NeuralNetwork
from functions import sigmoid
import numpy as np


if __name__ == '__main__':
    data = dict(np.load('tests/data.npz').items())

    weights = (data['theta1'], data['theta2'])
    numbers = data['numbers']

    nn = NeuralNetwork(weights)
    result = nn.predict(numbers[0])

    print(result)
    print(np.argmax(result)+1)