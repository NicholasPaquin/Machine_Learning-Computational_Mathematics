from computational_mathmatics import Node, Graph
from optim import OptimOld, Cost, Sigmoid
import numpy as np

class _Layer:
    """
    Layer is made up of nodes, either fully connected or not.
    idea of operation is to go by layer and each layer will preform and store the values from each node
    which is passed onto the next layer etc.
    The layer will all operate on the same activation function. Or series of activation functions.
    Specify width of layers, whether it is fully connected or not.
    When adding the next layer to the model, connect each node based on whether it is fully connected or not.
    """

    def __init__(self, input_features, output_features, input_layer=False, fully_connected=True):
        self.output_features = output_features
        self.fully_connected = fully_connected
        # changed from v0.0.0 #
        self.weights = np.random.rand(output_features, input_features)
        self.bias = np.random.randn(output_features)
        # last part for emphasis #
        self.next_layer = None
        self.prev_layer = None
        self.input_layer = input_layer
        self.variables = 0

    def eval(self, inputs: np.array, weights, bias):
        return inputs

    def activation(self, z):
        return z

    def initialize_layer(self, variables=0):
        if self.input_layer:
            self.variables = self.width
        else:
            self.variables = variables
        # self.initialize_weights(self.prev_layer)

    def connect(self, layer):
        self.next_layer = layer
        layer.prev_layer = self
        layer.initialize_weights(self.width)

    # model forward, passes through entire model
    def forward(self, inputs):
        if self.prev_layer:
            if self.prev_layer.width == 1:
                pass
            else:
                assert(inputs.size == self.prev_layer.width)
        else:
            assert (inputs.size == self.width)

        # testing for pure matrix algebra

        # zs = self.eval(inputs, self.weights, self.bias)
        # activations = self.activation(zs)

        activations = np.array([])
        zs = np.array([])
        for i in range(self.width):
            # evaluates the node based on the node type or the function specified
            z = self.eval(inputs, self.weights[i], self.bias[i])
            val = self.activation(z)
            zs = np.append(zs, z)
            activations = np.append(activations, val)
            activations = np.squeeze(activations)
        # function return the raw calculations to avoid repetition in back propogation
        return activations, zs

    def definition(self):
        # print(f"Width: {self.width}, Function: {self.node}, Fully Connected:"
        #       f" {self.fully_connected}, Sample Node: {self.nodes[0].node_def()}")
        pass


class Linear(_Layer):
    def eval(self, inputs: np.array, weights, bias):
        return np.dot(weights, inputs) + bias


class Sigmoid(Linear):
    def activation(self, z):
        return 1/(1 + np.exp(-z))


class Model:
    def __init__(self, layers: list, cost: Cost):
        print("Initialized model")
        self.weights = []
        self.bias = []
        self.layers = layers
        self.depth = len(layers)
        self.initialize_layers()
        self.optim = OptimOld(self, cost)
        # call generate graph function
        for i in range(0, self.depth - 1):
            self.layers[i].connect(self.layers[i+1])

    def init_input(self):
        self.layers[0].prev_layer = None
        self.input_layer = self.layers[0]
        # below used to initialize the weights for itself
        self.layers[0].input_layer = True
        self.layers[0].initialize_weights()

    def init_output(self):
        self.layers[-1].next_layer = None
        self.output_layer = self.layers[-1]

    def initialize_layers(self):
        self.init_input()
        self.init_output()
        for i in range(1, len(self.layers), 1):
            self.layers[i].initialize_layer(self.layers[i - 1].width)
        for layer in self.layers:
            self.bias.append(layer.bias)
            self.weights.append(layer.weights)

    # moves through each input and passes it through each node,
    # these values are then stored and passed along to the next nodes
    def forward(self, inputs: np.array):
        activation = inputs
        activations = np.array([inputs])
        zs = []
        for layer in self.layers:
            activation, z = layer.forward(activation)
            activations = np.append(activations, activation)
            zs = np.append(zs, z)
        return activations, zs

    # test function will likely br gone later
    # def feedforward(self, a):
    #     """Return the output of the network if ``a`` is input."""
    #     for b, w in zip(self.bias, self.weights):
    #         a = Sigmoid().eval(np.dot(w, a)+b)
    #     return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(x, np.argmax(self.predict(x)), y)
                        for (x, y) in test_data]
        print(test_results)
        return sum(int(x == y) for (z, x, y) in test_results)

    def predict(self, inputs: np.array):
        activation = inputs
        for layer in self.layers:
            activation, z = layer.forward(activation)
        return activation

    def details(self):
        for layer in self.layers:
            layer.definition()


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




