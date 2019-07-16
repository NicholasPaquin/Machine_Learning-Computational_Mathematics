from computational_mathmatics import Node, Graph


class Layer:
    """
    Layer is made up of nodes, either fully connected or not.
    idea of operation is to go by layer and each layer will preform and store the values from each node
    which is passed onto the next layer etc.
    The layer will all operate on the same activation function. Or series of activation functions.
    Specify width of layers, whether it is fully connected or not.
    When adding the next layer to the model, connect each node based on whether it is fully connected or not.
    """

    def __init__(self, width, function, input_layer=False, fully_connected=True):
        self.width = width
        self.function = function
        self.fully_connected = fully_connected
        self.nodes = []
        self.next_layer = None
        self.prev_layer = None

    def initialize_layer(self, variables):
        for i in range(self.width):
            self.nodes.append(Node(variables, self.function))

    def connect(self, layer):
        self.next_layer = layer
        layer.prev_layer = self
        for node in layer.nodes:
            node.connect(self.nodes)


class Model:
    def __init__(self, layers: list):
        self.layers = layers
        self.depth = len(layers)
        self.init_input()
        self.init_output()
        for i in range(0, self.depth - 1):
            self.layers[i].connect(self.layers[i+1])

    def init_input(self):
        self.layers[0].prev_layer = None
        self.input_layer = self.layers[0]

    def init_output(self):
        self.layers[-1].next_layer = None
        self.output_layer = self.layers[-1]

    # moves through each input and passes it through each node,
    # these values are then stored and passed along to the next nodes
    def forward(self, inputs):
        c_val = Values(inputs, self.layers[0])
        for layer in self.layers:
            n_val = Values()
            for node in layer.nodes:
                val, next = node.forward(c_val[node.uuid])  # create a bunch of graphs from the end????
                n_val.insert(val, next)
            c_val.copy(n_val)


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




