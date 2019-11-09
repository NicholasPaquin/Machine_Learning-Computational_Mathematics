import numpy as np
import random
from activations import sigmoid, sigmoid_derivative


class SGD:
    def __init__(self, model, cost, regularizer=None, gamma=1):
        self.model = model
        self.cost = cost(regularizer, gamma)
        self.log = []
        self.bpcost = BackpropCost(regularizer, gamma)

    def SGD(self, train, epochs, mini_batch_size, learnin_rate, validation=None):
        if len(validation) != 0:
            n_test = len(validation)
        n = len(train)
        for epoch in range(epochs):
            costs = []
            np.random.shuffle(train)
            mini_batches = [train[i:i + mini_batch_size] for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                cost = self.update_batch(mini_batch, learnin_rate)
                costs.append(cost)
            if len(validation) != 0:
                print(f"Epoch {epoch}: {self.model.evaluate(validation)} / {n_test},"
                      f" Average Cost: {np.average(np.array(costs))}")
            else:
                print(f"Epoch {epoch} Average Cost: {np.average(np.array(costs))}")

    def update_batch(self, batch, lr):
        nabla_b = [np.zeros(b.shape) for b in self.model.bias]
        nabla_w = [np.zeros(w.shape) for w in self.model.weights]
        costs = []
        for x, y in batch:
            delta_nabla_b, delta_nabla_w, cost = self.backprop(x, y)
            # print("nable_b")
            # print('len: ', len(delta_nabla_b))
            # print(delta_nabla_b)
            # print("nable_w")
            # print('len: ', len(delta_nabla_w))
            # print(delta_nabla_w)
            costs.append(cost)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.model.weights = [w - (lr / len(batch)) * nw for w, nw in zip(self.model.weights, nabla_w)]
        self.model.bias = [b - (lr / len(batch)) * nb for b, nb in zip(self.model.bias, nabla_b)]
        return np.average(np.array(costs))

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.model.bias]
        nabla_w = [np.zeros(w.shape) for w in self.model.weights]
        # activations, zs = self.model.forward(x)
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.model.bias, self.model.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        cost = self.cost.eval(activations[-1], y, self.model.weights)
        # delta = self.cost.derivative(activations[-1], y, zs[-1])
        nabla_b[-1] = delta = self.cost.delta_b(activations[-1], y, self.model.weights)
        nabla_w[-1] = self.cost.delta_w(activations[-1], activations[-2], y, self.model.weights)
        for l in range(2, self.model.depth + 1):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            # delta = np.dot(self.model.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w, cost


class Cost:
    @staticmethod
    def eval(predicted, expected):
        return np.power((predicted - expected), 2)

    @staticmethod
    def derivative(predicted, expected, z, x):
        return predicted - expected


class Quadratic:
    @staticmethod
    def eval(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def derivative(a, y, z):
        return (a-y) * sigmoid.derivative(z)


class LogLoss:
    def __init__(self, regularizer, gamma=1):
        self.gamma = gamma
        if regularizer == None:
            self.eval = self._eval
            self.derivative = self._derivative
        elif regularizer == 'l2':
            pass

    def _eval(self, predicted, expected, weights):
        if expected == 1.0:
            return -np.log(predicted)
        elif expected == 0.0:
            return -np.log(1 - predicted)

    def _derivative(self, predicted, expected, weights):
        if expected == 1.0:
            return -1 / predicted[1]
        elif expected == 0.0:
            return 1 / (1 - predicted[0])


class CrossEntropy:
    def __init__(self, regularizer=None, gamma=1):
        self.gamma = gamma
        if regularizer == None:
            self.eval = self._eval
            self.delta_w = self._delta_w
            self.delta_b = self._delta_b
        elif regularizer == 'l2':
            self.reg = L2()
            self.eval = self._eval_reg
            self.delta_w = self._delta_w_reg
            self.delta_b = self._delta_b_reg
        else:
            raise ValueError(f'{regularizer} is not a supported regularizer')

    def _eval(self, a, y, weights):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    def _delta_w(self, a, a_2, y, weights):
        # print(len(z))
        return np.dot(a - y, a_2.T)

    def _delta_b(self, a, y, weights):
        # print(len(z))
        return a - y

    def _eval_reg(self, a, y, weights):
        return self._eval(a, y, weights) + self.reg.eval(self.gamma, len(y), weights)

    def _delta_w_reg(self, a, a_2, y, weights):
        # print(len(z))
        return self._delta_w(a, a_2, y, weights) + self.reg.delta_w(self.gamma, len(y), weights)

    def _delta_b_reg(self, a, y, weights):
        # print(len(z))
        return self._delta_b(a, y, weights) + self.reg.delta_b(self.gamma, len(y), weights)


class L2:
    def eval(self, gamma, n, weights):
        return gamma/n * np.sum(np.square(weights))

    def delta_w(self, gamma, n, weights):
        return gamma/n * weights

    def delta_b(self, gamma, n, weights):
        return 0


class BackpropCost:
    def __init__(self, regularizer=None, gamma=1):
        self.gamma = gamma
        if regularizer == None:
            self.delta_w = self._delta_w
            self.delta_b = self._delta_b
        elif regularizer == 'l2':
            self.reg = L2()
            self.delta_w = self._delta_w_reg
            self.delta_b = self._delta_b_reg
        else:
            raise ValueError(f'{regularizer} is not a supported regularizer')

    def _delta_w(self, a, delta_b, weights):
        # print(len(z))
        return np.dot(delta_b, a.T)

    def _delta_b(self, weights, delta, ap):
        return np.dot(weights.T, delta) * ap

    def _delta_w_reg(self, a, delta_b, weights, weights_whole):
        # print(len(z))
        return self._delta_w(a, delta_b, weights) + self.reg.delta_w(self.gamma, len(a), weights_whole)

    def _delta_b_reg(self, weights, delta, ap, weights_whole):
        # print(len(z))
        return self._delta_b(weights, delta, ap) + self.reg.delta_b(self.gamma, len(ap), weights)