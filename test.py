from machine_learning import Model, Layer  # Perceptron, Sigmoid
from operations import Basic
from optim import SGD, LogLoss, Cost
import numpy as np
import pandas as pd
from benchmark import Network

layer1 = Layer(1, input_layer=True)
layer2 = Layer(100)
# layer3 = Layer(100)
# layer4 = Layer(30)
layer5 = Layer(1)

layers = [layer1, layer2, layer5]

model = Model(layers, LogLoss)
model.details()

# print(layer2.nodes[0].weights)
x = np.array([i for i in range(10001)])/1000
y = np.array([1 if elem >= 5 else 0 for elem in x])
data = pd.DataFrame([x, y]).T.to_numpy()
print(data)
# model.forward(a)
optim = SGD(model, LogLoss)
weights1 = optim.model.weights
bias1 = optim.model.bias
optim.SGD(data, 20, 100, 100, validation=data)
weights2 = optim.model.weights
bias2 = optim.model.bias


x = np.array([i for i in range(10001)])/1000
y = np.array([elem >= 5 for elem in x])
data = np.array(np.vstack((x, y)).T)

net = Network([1, 30, 2])
net.SGD(data, 1000, 10, 3.0, test_data=data)


# node1 = Node(1, Basic.assign)
# node2 = Node(1, Basic.assign)
# adder = Node(2, Basic.add)
# adder.connect([node1, node2])
# print(adder())
# graph = Graph(adder)
# graph.forward(np.array([10, 8]))
