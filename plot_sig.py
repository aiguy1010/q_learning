import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
    return  1 / (1 + np.exp(-x))

X = np.arange(-5, 5, 0.1)
Y = sigmoid(X)

plt.plot(X, Y)
plt.show()
