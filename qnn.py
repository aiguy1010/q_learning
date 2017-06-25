import numpy as np

FOOD = 1
SPIKES = -1
WALL = -5

# Construct Network
layers = []
W1 = np.random.normal(size=[25, 10])
b1 = np.random.normal(size=[10])
layers.append((W1, b1, 'sig'))
W2 = np.random.normal(size=[10, 4])
b2 = np.random.normal(size=[4])
layers.append((W2, b2, 'none'))
layer_input_cache = []

def infer(inputs):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    layer_input_cache = []
    last_outputs = inputs
    for l, (W, b) in enumerate(layers):
        layer_input_cache.append(last_outputs)
        partial_result = np.matmul(last_outputs, layer[0]) + layer[1]
        if l != len(layers)-1:
            last_outputs = sigmoid(partial_result)

    return last_outputs

def backprop(output_node, error, learning_rate):
    for W, b, activation in layers[::-1]:
