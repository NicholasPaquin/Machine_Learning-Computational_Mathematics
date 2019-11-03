import numpy as np


class Activation:
    @staticmethod
    def __call__(z):
        return z

    @staticmethod
    def derivative(z):
        return z


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))