import numpy as np

class Dropout:
    def __init__(self, model, p):
        self.model = model
        # should be a fraction between 0.5 and 1
        self.p = p
        self.dropout = []

    def __call__(self):
        layers = [len(weight) for weight in self.model.bias[:-1]]
        self.dropout = [np.sort(np.random.choice(layer, int(np.ceil(layer * (1 - self.p))), replace=False)) for layer in layers]

