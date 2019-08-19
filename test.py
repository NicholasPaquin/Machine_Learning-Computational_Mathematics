from machine_learning import Model, Layer, Perceptron, Sigmoid
from operations import Basic
from optim import Optim, Cost
import numpy as np

layer1 = Layer(1, Sigmoid)
layer2 = Layer(5, Sigmoid)
layer3 = Layer(1, Sigmoid)

layers = [layer1, layer2, layer3]

model = Model(layers, Cost)
model.details()

# print(layer2.nodes[0].weights)
a = np.random.rand(100, 2)
# model.forward(a)

optim = Optim(model, Cost)
optim.SGD(a, 1, 10, 0.01)

# node1 = Node(1, Basic.assign)
# node2 = Node(1, Basic.assign)
# adder = Node(2, Basic.add)
# adder.connect([node1, node2])
# print(adder())
# graph = Graph(adder)
# graph.forward(np.array([10, 8]))
