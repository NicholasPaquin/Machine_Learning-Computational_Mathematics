from machine_learning import Model, Layer
from operations import Basic

layer1 = Layer(100, Basic.assign)
layer2 = Layer(100, Basic.assign)
layer3 = Layer(100, Basic.assign)

# layer1.initialize_layer(100)
# layer2.initialize_layer(100)
# layer3.initialize_layer(100)

layers = [layer1, layer2, layer3]

model = Model(layers)
model.details()