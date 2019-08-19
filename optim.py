import numpy as np
import random


class Optim:
    def __init__(self, model, cost):
        self.model = model
        self.cost = cost

    def SGD(self, train, epochs, mini_batch_size, learnin_rate, validation=None):
        n = len(train)
        for i in range(epochs):
            random.shuffle(train)
            mini_batches = [train[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.evaluate_batch(mini_batch, learnin_rate)

    def evaluate_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.model.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.model.layers]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        for layer in self.model.layers:
            layer.weights = [w - (learning_rate / len(batch)) * nw for w, nw in zip(layer.weights, nabla_w)]
            layer.biases = [b - (learning_rate / len(batch)) * nb for b, nb in zip(layer.biases, nabla_b)]

    def backprop(self, x, y):
        print(x)
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.model.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.model.layers]
        activation_prime = self.model.layers[-1].node.derivative

        # forward pass
        activation = x
        activations = np.array([x])
        zs = []
        for layer in self.model.layers:
            layer.definition()
            activation, z = layer.forward(activation)
            activations = np.append(activations, activation)
            zs = np.append(zs, z)

        delta = self.cost.derivative(activations[-1], y) * activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].transpose(), delta)

        for l in range(2, len(self.model.layers)):
            activation_prime = self.model.layers[-l].node.derivative
            z = zs[-l]
            ap = activation_prime(z)
            delta = np.dot(delta, self.model.layers[-l + 1].weights.transpose()) * ap
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(activations[-l - 1], delta)
        return (nabla_b, nabla_w)

class Cost:
    def __call__(self, inputs):
        pass

    def derivative(self, output_activations, y):
        return output_activations - y



