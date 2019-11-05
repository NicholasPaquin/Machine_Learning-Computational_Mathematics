import numpy as np


class Activation:
    @staticmethod
    def __call__(z):
        return z

    @staticmethod
    def derivative(z):
        return z


class _Sigmoid(Activation):
    def __call__(self, z):
        return sigmoid(z)

    def derivative(self, z):
        return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))