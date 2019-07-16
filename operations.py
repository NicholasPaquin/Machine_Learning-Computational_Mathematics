def assign(val):
    return val


def add(vals):
    sum = 0
    for i in vals:
        sum += i
    return sum


class ActivationFunctions:
    def relu(self, weight):
        return weight if weight > 0 else 0

    def sigmoid(self, weight):
        pass

    def l_relu(self, weight, slope=0.1):
        return weight if weight > 0 else weight*slope

    def tanh(self, weight):
        pass

    def softmax(self, weight):
        pass


