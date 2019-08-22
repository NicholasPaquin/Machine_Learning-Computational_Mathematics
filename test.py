from machine_learning import Model, Layer, Perceptron, Sigmoid
from operations import Basic
from optim import Optim, Cost
import numpy as np
import pandas as pd

layer1 = Layer(1, Sigmoid)
layer2 = Layer(128, Sigmoid)
layer3 = Layer(256, Sigmoid)
layer4 = Layer(128, Sigmoid)
layer5 = Layer(1, Sigmoid)

layers = [layer1, layer2, layer3, layer4, layer5]

model = Model(layers, Cost)
model.details()

# print(layer2.nodes[0].weights)
x = np.random.randn(10000)
y = x*3
data = np.array(np.vstack((x, y)).T)
print(data)
# model.forward(a)

optim = Optim(model, Cost)
optim.SGD(data, 100, 10, 0.001)
print("----------------------")

print(model.forward(data[0][0]))
print(data[0][1])

# node1 = Node(1, Basic.assign)
# node2 = Node(1, Basic.assign)
# adder = Node(2, Basic.add)
# adder.connect([node1, node2])
# print(adder())
# graph = Graph(adder)
# graph.forward(np.array([10, 8]))
