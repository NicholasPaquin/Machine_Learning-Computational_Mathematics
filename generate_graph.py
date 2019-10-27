"""
This document is used to dynamically generate graphs that are specifically designed for my machine learning library
"""


class NNGraph:
    def __init__(self, model):
        # becomes too heavy if it is carrying around parameters i think for now I'll keep and check performance later
        self.parameters = model._parameters()
        self.length = len(self.parameters)
        self.start_node = None
        self.end_node = None
        self.generate_forward()
        self.weights = model.weights
        self.bias = model.bias

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

    def generate_backwards(self):
        count = 0
        for param in self.parameters:
            node = NNBackward(param.in_features, param.out_features, derivative, count)
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
        pass


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
    def __init__(self, in_features, out_features, id, derivative):
        super().__init__(in_features, out_features, id)
        self.derivative = derivative

