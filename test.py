from machine_learning import Model, Sigmoid  # Perceptron, Sigmoid
from operations import Basic
from optim import SGD, LogLoss, Cost
import numpy as np
import pandas as pd
from benchmark import Network


x = np.array([i for i in range(10001)])/1000
y = np.array([elem >= 5 for elem in x])
data = np.array(np.vstack((x, y)).T)

net = Network([1, 100, 32, 2])
net.SGD(data, 100, 10, 3.0, test_data=data)



