from machine_learning import Model, Layer, Perceptron, Sigmoid
from operations import Basic
from optim import Optim, Cost
import numpy as np

layer1 = Layer(1, Sigmoid)
layer2 = Layer(32, Sigmoid)
layer3 = Layer(64, Sigmoid)
layer4 = Layer(32, Sigmoid)
layer5 = Layer(1, Sigmoid)

layers = [layer1, layer2, layer3, layer4, layer5]

model = Model(layers, Cost)
model.details()

# print(layer2.nodes[0].weights)
a = np.random.randn(10000, 2)
# model.forward(a)

optim = Optim(model, Cost)
optim.SGD(a, 10, 10, 0.0001)
print("----------------------")

print(model.forward(a[0][0]))
print(a[0][1])

# node1 = Node(1, Basic.assign)
# node2 = Node(1, Basic.assign)
# adder = Node(2, Basic.add)
# adder.connect([node1, node2])
# print(adder())
# graph = Graph(adder)
# graph.forward(np.array([10, 8]))
