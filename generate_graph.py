"""
This document is used to dynamically generate graphs that are specifically designed for my machine learning library
"""
import numpy as np


class NNGraphForward:
    def __init__(self, model):
        # becomes too heavy if it is carrying around parameters i think for now I'll keep and check performance later
        self.parameters = model._parameters()
        self.length = len(self.parameters)
        self.start_node = None
        self.end_node = None
        self.generate_forward()
        self.model = model
        # self.weights = model.weights
        # self.bias = model.bias

    def __call__(self, *args, **kwargs):
        pass

    def generate_forward(self):
        count = 0
        for param in self.parameters:
            node = NNForward(param.in_features, param.out_features, param.function, param.activation, count)
            if count == 0:
                self.start_node = node
            elif count == self.length:
                self.end_node = node
                self.insert_node(node)
            else:
                self.insert_node(node)

            count += 1

    def insert_node(self, node):
        self._insert_node(node, self.start_node, None)

    def _insert_node(self, insert_node, current_node, prev_node):
        if current_node.id < insert_node.id \
                and ((not current_node.next) or (current_node.next and insert_node.id < current_node.next.id)):
            insert_node.next = current_node.next
            insert_node.prev = current_node
            current_node.next = insert_node
            current_node.prev = prev_node
        else:
            self._insert_node(insert_node, current_node.next, current_node)

    def traverse(self):
        node = self.start_node
        while node:
            print(f'{type(node)} {node.id}: in_features: {node.in_features},'
                  f' out_features: {node.out_features}')
            node = node.next

    def forward(self, x):
        activation = x
        activations = [x]
        zs = []
        layer = 0
        node = self.start_node
        for b, w in zip(self.model.bias, self.model.weights):
            if self.model.dropout and layer != self.length - 1:
                w = w * self.model.dropout.p
            z = np.dot(w, activation)+b
            activation = node.activation(z)
            activations.append(activation)
            zs.append(z)
            layer += 1
        return activations, zs

    def predict(self, x):
        return self.forward(x)[0][-1]



class NNGraphBackward:
    def __init__(self, model):
        # becomes too heavy if it is carrying around parameters i think for now I'll keep and check performance later
        self.parameters = model._parameters()
        self.length = len(self.parameters)
        self.start_node = None
        self.end_node = None
        self.generate_backward()
        self.model = model

    def __call__(self, *args, **kwargs):
        pass

    def generate_backward(self):
        for count, param in enumerate(self.parameters[::-1]):
            node = NNBackward(param.in_features, param.out_features, param.activation.derivative, count)
            if count == 0:
                self.start_node = node
            elif count == self.length:
                self.end_node = node
                self.insert_node(node)
            else:
                self.insert_node(node)

    def insert_node(self, node):
        self._insert_node(node, self.start_node, None)

    def _insert_node(self, insert_node, current_node, prev_node):
        if current_node.id < insert_node.id \
                and ((not current_node.next) or (current_node.next and insert_node.id < current_node.next.id)):
            insert_node.next = current_node.next
            insert_node.prev = current_node
            current_node.next = insert_node
            current_node.prev = prev_node
        else:
            self._insert_node(insert_node, current_node.next, current_node)

    def traverse(self):
        node = self.start_node
        while node:
            print(f'{type(node)} {node.id}: in_features: {node.in_features},'
                  f' out_features: {node.out_features}')
            node = node.next

    def backprop(self, x, y, n, cost, bpcost):
        nabla_b = [np.zeros(b.shape) for b in self.model.bias]
        nabla_w = [np.zeros(w.shape) for w in self.model.weights]
        activations, zs = self.model.forward(x)
        cost_score = cost.eval(activations[-1], y, self.model.weights, n)
        # delta = self.cost.derivative(activations[-1], y, zs[-1])
        node = self.start_node
        nabla_b[-1] = delta = cost.delta_b(activations[-1], y, self.model.weights[-1], n)
        nabla_w[-1] = cost.delta_w(activations[-1], activations[-2], y, self.model.weights[-1], n)
        node = node.next
        for l in range(2, self.model.depth + 1):
            z = zs[-l]
            ap = node.derivative(z)
            # delta = np.dot(self.model.weights[-l + 1].T, delta) * sp
            delta = bpcost.delta_b(self.model.weights[-l + 1], delta, ap, self.model.weights, n)
            nabla_b[-l] = delta
            # np.dot(delta, activations[-l - 1].T)
            nabla_w[-l] = bpcost.delta_w(activations[-l - 1], delta, self.model.weights[-l], self.model.weights, n)
        return nabla_b, nabla_w, cost_score

class _NNNode:
    def __init__(self, in_features, out_features, id):
        self.next = None
        self.prev = None
        self.in_features = in_features
        self.out_features = out_features
        self.id = id

class NNForward(_NNNode):
    def __init__(self, in_features, out_features, function, activation, id):
        super().__init__(in_features, out_features, id)
        self.function = function
        self.activation = activation


class NNBackward(_NNNode):
    def __init__(self, in_features, out_features, derivative, id):
        super().__init__(in_features, out_features, id)
        self.derivative = derivative

