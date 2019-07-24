import numpy as np
import random


class Optim:
    def __init__(self, model):
        self.model = model

    def SGD(self, train, epochs, mini_batch_size, learnin_rate, validation=None):
        n = len(train)
        for i in range(epochs):
            random.shuffle(train)
            mini_batches = [train[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.evaluate_batch(mini_batch, learnin_rate)

    def evaluate_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(layer.shape("b")) for layer in self.model.layers]
        nabla_w = [np.zeros(layer.shape("w")) for layer in self.model.layers]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # Your code sucks
        pass

