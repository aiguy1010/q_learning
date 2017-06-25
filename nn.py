import numpy as np

# First iteration will be a somewhat hard coded 3-bit binary counter
def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x) / (1+np.exp(-x))**2

# Construct Network
layers = []
W1 = np.random.normal(size=[3, 3])
b1 = np.random.normal(size=[3])
layers.append((W1, b1, 'sig'))
W2 = np.random.normal(size=[3, 3])
b2 = np.random.normal(size=[3])
layers.append((W2, b2, 'sig'))
excitements_cache = []


def infer(X):
    excitements_cache = [X]
    act_fn = None
    for l, (W, b, activation) in enumerate(layers):
        if activation is None or activation == 'none'
            act_fn = lambda x:x
        elif activation == 'sig':
            act_fn = sigmoid

        z = np.matmul(act_fn(excitements_cache[-1]), W) + b
        excitements_cache.append( z )

    return act_fn(z)

def backprop(target, learning_rate):
    error = activations_cache[-1] - target
    for l, (W, b, activation) in enumerate(layers[::-1]):
        dW = np.matmul(np.transpose(layers[::-1][l+1]),  )
