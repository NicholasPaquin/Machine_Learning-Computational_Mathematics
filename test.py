from machine_learning import Model, Layer, Perceptron
from operations import Basic

layer1 = Layer(100, Basic.assign, Perceptron)
layer2 = Layer(100, Basic.assign, Perceptron)
layer3 = Layer(100, Basic.assign, Perceptron)

# layer1.initialize_layer(100)
# layer2.initialize_layer(100)
# layer3.initialize_layer(100)

layers = [layer1, layer2, layer3]

model = Model(layers)
model.details()

print(layer1.nodes[0].weights)



# node1 = Node(1, Basic.assign)
# node2 = Node(1, Basic.assign)
# adder = Node(2, Basic.add)
# adder.connect([node1, node2])
# print(adder())
# graph = Graph(adder)
# graph.forward(np.array([10, 8]))
