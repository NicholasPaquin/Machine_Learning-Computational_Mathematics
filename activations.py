import numpy as np


class Activation:
    @staticmethod
    def __call__(z):
        return z

    @staticmethod
    def derivative(z):
        return z


class sigmoid(Activation):
    @staticmethod
    def __call__(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        eval = sigmoid()
        return eval(z) * (1 - eval(z))
