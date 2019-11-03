from generate_graph import NNGraph
from optim import Cost
import numpy as np
from base import ModelParameters
from activations import sigmoid, sigmoid_derivative


class Model:
    def __init__(self, layers: list, cost: Cost, optim):
        print("Initialized model")
        self.weights = []
        self.bias = []
        self.layers = layers
        self.depth = len(layers)
        self.initialize_layers()
        self._optim = optim(self, cost)
        self.input_layer = None
        self.output_layer = None
        # call generate graph function
        for i in range(0, self.depth - 1):
            self.layers[i].connect(self.layers[i+1])
        self._parameters = ModelParameters(self)
        self._graph = NNGraph(self)

    def init_input(self):
        self.layers[0].prev_layer = None
        self.input_layer = self.layers[0]
        # below used to initialize the weights for itself
        self.layers[0].input_layer = True

    def init_output(self):
        self.layers[-1].next_layer = None
        self.output_layer = self.layers[-1]

    def initialize_layers(self):
        self.init_input()
        self.init_output()
        for layer in self.layers:
            self.bias.append(layer.bias.reshape(layer.bias.shape[0], 1))
            self.weights.append(layer.weights)

    # moves through each input and passes it through each node,
    # these values are then stored and passed along to the next nodes
    def forward(self, inputs: np.array):
        activation = inputs
        activations = [inputs]
        zs = []
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation)+b
            activation = sigmoid(z)
            activations.append(activation)
            zs.append(z)

        return activations, zs

    # # test function will likely br gone later
    # def feedforward(self, a):
    #     """Return the output of the network if ``a`` is input."""
    #     for b, w in zip(self.bias, self.weights):
    #         a = Sigmoid()(np.dot(w, a)+b)
    #     return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(x, np.argmax(self.predict(x, False)), y)
                        for (x, y) in test_data]
        # print(test_results)
        return sum(int(x == y) for (z, x, y) in test_results)

    def predict(self, inputs: np.array, test):
        activation = inputs
        for b, w in zip(self.bias, self.weights):
            z = np.dot(w, activation) + b
            if test:
                print(z)
            activation = sigmoid(z)
            if test:
                print(activation)
                print(f"activation check: {not z == activation}")
        # print(activation)
        return activation

    def parameters(self):
        print('----------Model Parameters----------')
        for param in self._parameters():
            print(f'{param.type}: in_features: {param.in_features} '
                  f'out_features: {param.out_features}')
        print('------------------------------------')


# stores values to be processed by layers, uses dictionary and sorts by node UUID
class Values:
    def __init__(self, inputs=None, nodes=None):
        assert(len(inputs) == len(nodes))
        self.val = {}
        if inputs and nodes:
            for i in range(len(inputs)):
                self.insert(inputs[i], nodes[i].uuid)

    def __getitem__(self, item):
        return self.val[item]

    def __iter__(self):
        if hasattr(self.val[0], "__iter__"):
            return self.val[0].__iter__()
        return self.val.__iter__()

    def __len__(self):
        return len(self.val)

    def insert(self, input, node):
        if node in self.val:
            self.val[node].extend([input])
        else:
            self.val[node] = [input]

    def copy(self, values):
        self.val = values.val




