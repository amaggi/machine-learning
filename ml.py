from functions import sigmoid
import numpy as np

class NeuralNetwork(object):
    def __init__(self, weights, activation_function=sigmoid):
        self.weights = weights
        self.h = activation_function

    def predict(self, input_layer, add_bias=True):
        output_layer = input_layer
        for weight in self.weights:
            if add_bias:
                output_layer = np.append(1, output_layer)

            output_layer = weight.dot(output_layer)

        return output_layer