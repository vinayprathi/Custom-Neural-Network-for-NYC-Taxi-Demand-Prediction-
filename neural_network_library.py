
import numpy as np

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, input) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(self.weights.T, output_gradient)
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        sigmoid_gradient = self.output * (1 - self.output)
        return output_gradient * sigmoid_gradient

class MeanSquaredError:
    def calculate_loss(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def gradient(self, predicted, actual):
        return 2 * (predicted - actual) / len(predicted)

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_gradient, learning_rate):
        gradient = output_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
        return gradient
