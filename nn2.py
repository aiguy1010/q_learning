import numpy as np

class NeuralNetwork:
    def __init__(self, layer_info, input_size):
        self.layer_info = layer_info

        # Initialize weights and biases (weights are associated with the units above them)
        self.layers = []
        last_units = input_size
        for units, activation_str in self.layer_info:
            W = np.random.normal(size=[last_units, units])
            b = np.random.normal(size=[units])
            self.layers.append((W, b, activation_str))

        self.activations_cache = []
        self.activations_fns = {
            'sig': lambda x:1/(1+np.exp(-x))
            'sig_prime': lambda x:np.exp(-x)/(1+np.exp(-x))**2
        }

    def infer(X):
        self.activations_cache = [X]
        for W, b, activation_str in self.layers:
            prev_outs = activations_cache[-1]
            if activation_str == 'sig':
                for
            next_outs = np.matmul(prev_outs, W) + b
            self.activations_cache.append(next_outs)
        return activations_cache[-1]

if __name__ == '__main__':
    X = [1, 0]
