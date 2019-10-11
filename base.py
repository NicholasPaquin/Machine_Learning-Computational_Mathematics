class ModelParameters:
    def __init__(self, model):
        self.parameters = [LayerParameter(layer) for layer in model.layers]


class LayerParameter:
    def __init__(self, layer):
        self.type = type(layer)
        self.function = layer.eval
        self.activation = layer.activation
        self.output_features = layer.output_features
        self.input_features = layer.input_features
