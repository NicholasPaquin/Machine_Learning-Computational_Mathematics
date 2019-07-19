from machine_learning import Model, Layer, Perceptron
from operations import Basic
import numpy as np

layer1 = Layer(100, Perceptron)
layer2 = Layer(100, Perceptron)
layer3 = Layer(100, Perceptron)

# layer1.initialize_layer(100)
# layer2.initialize_layer(100)
# layer3.initialize_layer(100)

layers = [layer1, layer2, layer3]

model = Model(layers)
model.details()

print(layer2.nodes[0].weights)
a = np.zeros(100)
print(a.size)
print(layer2.prev_layer.width)
print(layer2.prev_layer.width == a.size)
layer2.forward(a)

# node1 = Node(1, Basic.assign)
# node2 = Node(1, Basic.assign)
# adder = Node(2, Basic.add)
# adder.connect([node1, node2])
# print(adder())
# graph = Graph(adder)
# graph.forward(np.array([10, 8]))
