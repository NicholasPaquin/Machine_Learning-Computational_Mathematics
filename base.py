class ModelParameters:
    def __init__(self, model):
        self.parameters = [LayerParameter(layer) for layer in model.layers]

    def __call__(self, *args, **kwargs):
        return self.parameters


class LayerParameter:
    def __init__(self, layer):
        self.type = type(layer)
        self.function = layer.eval
        self.activation = layer.activation
        self.out_features = layer.out_features
        self.in_features = layer.in_features
        self.weights = layer.weights
        self.bias = layer.bias
