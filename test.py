from machine_learning import Model, Layer, Perceptron, Sigmoid
from operations import Basic
import numpy as np

layer1 = Layer(100, Sigmoid)
layer2 = Layer(100, Sigmoid)
layer3 = Layer(100, Sigmoid)
layer4 = Layer(30, Sigmoid)
layer5 = Layer(10, Sigmoid)


layers = [layer1, layer2, layer3, layer4, layer5]

model = Model(layers)
model.details()

# print(layer2.nodes[0].weights)
a = np.ones(100)
model.forward(a)

# node1 = Node(1, Basic.assign)
# node2 = Node(1, Basic.assign)
# adder = Node(2, Basic.add)
# adder.connect([node1, node2])
# print(adder())
# graph = Graph(adder)
# graph.forward(np.array([10, 8]))
