import numpy as np
import random

# piece of shit doesn't work
class OptimOld:
    def __init__(self, model, cost):
        self.model = model
        self.cost = cost

    def SGD(self, train, epochs, mini_batch_size, learnin_rate, validation=None):
        n = len(train)
        for i in range(epochs):
            print(f"Epoch: {i+1}/{epochs}")
            random.shuffle(train)
            mini_batches = [train[i:i+mini_batch_size] for i in range(0, n, mini_batch_size)]
            costs = np.array([])
            for mini_batch in mini_batches:
                cost = self.evaluate_batch(mini_batch, learnin_rate)
                costs = np.append(costs, cost)
            print(f"Average cost: {np.average(costs)}")

    # change everything eventually to purely matrix operations
    def evaluate_batch(self, batch, learning_rate):
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.model.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.model.layers]
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = nabla_b + delta_nabla_b
            nabla_w = nabla_w + delta_nabla_w
        for i in range(len(self.model.layers)):
            self.model.layers[i].weights = self.model.layers[i].weights - (learning_rate / len(batch)) * nabla_w[i]
            self.model.layers[i].bias = self.model.layers[i].bias - (learning_rate/len(batch)) * nabla_b[i]
            # layer.weights = layer.weights - (learning_rate / len(batch)) * nabla_w
            # layer.biases = layer.bias - (learning_rate / len(batch)) * nabla_b
        return 1

    def backprop(self, x, y):
        # print(x)
        nabla_b = [np.zeros(layer.bias.shape) for layer in self.model.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.model.layers]
        activation_prime = self.model.layers[-1].nodes[0].derivative

        # forward pass
        activation = x
        activations = np.array([x])
        zs = []
        for layer in self.model.layers:
            activation, z = layer.forward(activation)
            activations = np.append(activations, activation)
            zs = np.append(zs, z)

        delta = self.cost.derivative(activations[-1], y) * activation_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, len(self.model.layers)):
            activation_prime = self.model.layers[-l].nodes[0].derivative
            z = zs[-l]
            ap = activation_prime(z)
            delta = np.dot(self.model.layers[-l + 1].weights.transpose(), delta) * ap
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(activations[-l - 1], delta)
        return nabla_b, nabla_w


class SGD:
    def __init__(self, model, cost):
        self.model = model
        self.cost = cost()
        self.log = []

    def SGD(self, train, epochs, mini_batch_size, learnin_rate, validation=[]):
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
            # print("---------------before epoch------------------")
            # print(train)
            if len(validation) != 0:
                print(f"Epoch {epoch}: {self.model.evaluate(validation)} / {n_test},"
                       f" Average Cost: {np.average(np.array(costs))}")
            else:
                print(f"Epoch {epoch} Average Cost: {np.average(np.array(costs))}")
            # print("---------------after epoch------------------")
            # print(train)

    def update_batch(self, batch, lr):
        nabla_b = [np.zeros(b.shape) for b in self.model.bias]
        nabla_w = [np.zeros(w.shape) for w in self.model.weights]
        costs = []
        for x, y in batch:
            delta_nabla_b, delta_nabla_w, cost = self.backprop(x, y)
            costs.append(cost)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # for i in range(len(self.model.layers)):
            #     nabla_b[i] = nabla_b[i] + np.squeeze(delta_nabla_b[i])
            #     nabla_w[i] = nabla_w[i] + delta_nabla_w[i]
        self.model.weights = [w - (lr / len(batch)) * nw for w, nw in zip(self.model.weights, nabla_w)]
        self.model.bias = [b - (lr / len(batch)) * nb for b, nb in zip(self.model.bias, nabla_b)]
        # count = 0
        # print("Updating weights and bias")
        # for layer in self.model.layers:
        #     # print('--------------------')
        #     # print(layer.weights.shape, layer.bias.shape, ' // ', nabla_w[count].shape, nabla_b[count].shape)
        #     layer.weights = layer.weights - lr/len(batch) * nabla_w[count]
        #     layer.bias = layer.bias - lr/len(batch) * nabla_b[count]
        #     # print(layer.weights.shape, layer.bias.shape)
        #     # print('--------------------')
        #     count += 1
        return np.average(np.array(costs))

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.model.bias]
        nabla_w = [np.zeros(w.shape) for w in self.model.weights]
        # activation = x
        # activations = [x]  # list to store all the activations, layer by layer
        # zs = []  # list to store all the z vectors, layer by layer
        # for b, w in zip(self.model.bias, self.model.weights):
        #     z = np.dot(w, activation) + b
        #     zs.append(z)
        #     activation = Sigmoid().eval(z)
        #     activations.append(activation)
        activations, zs = self.model.forward(x)
        # beware hard coding for testing
        cost = LogLoss().eval(activations[-1], y)
        # beware hardcoded for testing
        self.log.append(f"Cost; {cost}")
        self.log.append(f"Activation: {activations[-1]}, y: {y}, z = {zs[-1]}")
        self.log.append(f"Cost Derivative {self.cost.derivative(activations[-1], y)}, Sigmoid Prime {Sigmoid().derivative(zs[-1])}")
        delta = self.cost.derivative(activations[-1], y) * Sigmoid().derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.model.depth+1):
            z = zs[-l]
            sp = Sigmoid().derivative(z)
            delta = np.dot(self.model.layers[-l+1].weights.T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)
        return nabla_b, nabla_w, cost


class Cost:
    @staticmethod
    def eval(predicted, expected):
        pass

    @staticmethod
    def derivative(predicted, expected):
        return predicted - expected


class LogLoss(Cost):
    @staticmethod
    def eval(predicted, expected):
        if expected == 1.0:
            return -np.log(predicted)
        elif expected == 0.0:
            return -np.log(1-predicted)

    @staticmethod
    def derivative(predicted, expected):
        if expected == 1.0:
            return -1/predicted
        elif expected == 0.0:
            return 1/(1 - predicted)


class Sigmoid:
    @staticmethod
    def eval(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        return Sigmoid.eval(z) * (1 - Sigmoid.eval(z))

