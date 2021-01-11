import numpy as np
from activations import softmax, sigmoid


class RNNcell:
    def __init__(self, model):
        self.model = model
        self.a = 0
        self.y = 0

    def forward(self, x, a_prev):
        a_next = self.a = np.tanh(np.dot(self.model.w_aa, a_prev) + np.dot(self.model.w_ax, x) + self.model.b_a)
        y = self.y = softmax(np.dot(self.model.w_ya, a_next) + self.model.b_y)
        return a_next, y


class LSTMcell:
    def __init__(self):
        self.gamma_f = 0
        self.gamma_u = 0
        self.gamma_o = 0
        self.c_ = 0
        self.c = 0
        self.a = 0
        self.y = 0

    def forward(self, x, a_prev, c_prev):
        self.gamma_f = sigmoid(np.dot(self.w_f, np.concatenate([a_prev, x])) + self.b_f)
        self.gamma_u = sigmoid(np.dot(self.w_u, np.concatenate([a_prev, x])) + self.b_u)
        self.gamma_o = sigmoid(np.dot(self.w_o, np.concatenate([a_prev, x])) + self.b_o)
        self.c_ = np.tanh(self.w_c * np.concatenate([a_prev, x]) + self.b_c)
        self.c = self.gamma_f * c_prev + self.gamma_u * self.c_
        self.a = self.gamma_o * np.tanh(self.c)
        self.y = softmax(np.dot())


class LSTMlayer:
    def __init__(self):
        pass


class RNNlayer:
    def __init__(self, cells, model):
        pass

class LSTM:
    def __init__(self, n_a, n_x, n_y, m, cells, layers, random_seed=1):
        self.w_f = np.random.randn(n_a, n_a + n_x)
        self.w_u = np.random.randn(n_a, n_a + n_x)
        self.w_o = np.random.randn(n_a, n_a + n_x)
        self.w_c = np.random.randn(n_a, n_a + n_x)
        self.w_y = np.random.randn(n_y, n_a)
        self.b_f = np.random.randn(n_a, 1)
        self.b_u = np.random.randn(n_a, 1)
        self.b_o = np.random.randn(n_a, 1)
        self.b_c = np.random.randn(n_a, 1)
        self.b_y = np.random.randn(n_y, 1)


class RNN:
    def __init__(self, n_a, n_x, n_y, m, cells, layers, random_seed=1):
        self.n_a = n_a
        self.n_x = n_x
        self.n_y = n_y
        self.m = m
        self.T = cells
        self.cells = [[RNNcell(self) for i in range(cells)] for l in range(layers)]
        self.layers = layers
        np.random.seed(random_seed)
        self.w_aa = np.random.randn(n_a, n_a)
        self.w_ax = np.random.randn(n_a, n_x)
        self.w_ya = np.random.randn(n_y, n_a)
        self.b_a = np.random.randn(n_a, 1)
        self.b_y = np.random.randn(n_y, 1)
        self.a = np.zeros((self.n_a, self.m, self.T))
        self.y = np.zeros((self.layers, self.n_y, self.m, self.T))
        self.dx = 0
        self.da0 = 0

    def forward(self, x, a0):
        a_next = a0
        for l in range(self.layers):
            y = 0
            for t in range(self.T):
                a_next, y = self.cells[l][t].forward(x[:, :, t], a_next)
                self.a[:, :, t] = a_next
                self.y[l, :, :, t] = y
            x = y[l]

        return self.a, self.y

    def backward(self, x, da):
        dx = np.zeros((self.n_x, self.m, self.T))
        dw_ax = np.zeros((self.n_a, self.n_x))
        dw_aa = np.zeros((self.n_a, self.n_a))
        db_a = np.zeros((self.n_a, 1))
        da0 = np.zeros((self.n_a, self.m))
        da_prevt = np.zeros((self.n_a, self.m))
        for t in reversed(range(self.T)):
            dtanh = 1-self.a[:, :, t]**2 * (da[:, :, t] + da_prevt)
            dx[:, :, t] = np.dot(self.w_ax.T, dtanh)
            dw_ax += np.dot(dtanh, x[:, :, t].T)
            da_prevt = np.dot(self.w_aa.T, dtanh)
            dw_aa += np.dot(dtanh, self.a[:, :, t - 1].T)
            db_a += np.sum(dtanh, 1, keepdims=True)

        da0 = da_prevt
        self.dx = dx
        self.da0 = da0
        return dw_ax, dw_aa, db_a, dx, da0

    def update(self, dw_ax, dw_aa, db_a, dx, da0):