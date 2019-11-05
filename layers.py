import numpy as np


class _Layer:
    """
    things have changed...
    """

    def __init__(self, in_features, out_features, input_layer=False, fully_connected=True):
        self.in_features = in_features
        self.out_features = out_features
        self.fully_connected = fully_connected
        # changed from v0.0.0 #
        self.weights = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features)
        # last part for emphasis #
        self.next_layer = None
        self.prev_layer = None
        self.input_layer = input_layer
        self.variables = 0

    def eval(self, inputs: np.array, weights, bias):
        return inputs

    def activation(self, z):
        return z

    def connect(self, layer):
        self.next_layer = layer
        layer.prev_layer = self

    # model forward, passes through entire model
    def forward(self, inputs):

        assert(inputs.size == self.in_features)
        # testing for pure matrix algebra

        # zs = self.eval(inputs, self.weights, self.bias)
        # activations = self.activation(zs)

        activations = np.array([])
        zs = np.array([])
        for i in range(self.out_features):
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


class Conv2D(_Layer):
    def __init__(self, in_features, out_features, kernel_size, type='max', stride=1, padding=0, dilation=1):
        super().__init__(in_features, out_features)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def eval(self, inputs, weights, bias):
        pass
