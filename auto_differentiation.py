class AutoDif:
    def __init__(self, graph):
        self.graph = graph

    def differentiate(self):
        pass

    def evaluate(self, value):
        pass


class D:
    def __mul__(self, other):
        if type(other) == D:
            value = [0]
        else:
            value = [self, other]
        return value

    def __rmul__(self, other):
        if type(other) == D:
            value = [0]
        else:
            value = [self, other]
        return value
