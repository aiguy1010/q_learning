import numpy as np
import random

class NeuralNetwork:
    def __init__(self, layer_info, input_size):
        self.layer_info = layer_info

        # Initialize weights and biases (weights are associated with the units above them)
        self.layers = []
        last_units = input_size
        print(layer_info)
        for units, activation_str in self.layer_info:
            W = np.random.normal(size=[last_units, units])
            b = np.random.normal(size=[1, units])
            last_units = units
            self.layers.append((W, b, activation_str))

        self.z_cache = []
        self.activation_fns = {
            'identity': lambda x:x,
            'identity_prime': lambda x:1,
            'sig': lambda x:1/(1+np.exp(-x)),
            'sig_prime': lambda x:np.exp(-x)/(1+np.exp(-x))**2,
        }
        self.z_cache_stash = []


    def pushfront_z_cache(self):
        cache_copy = np.array()
        for layer_zs in self.z_cache:
            cache_copy.append( np.array(layer_zs) )
        self.z_cache_stash.insert(0, cache_copy)


    def popback_z_cache(self):
        item = self.z_cache_stash.pop()


    def infer(self, X):
        self.z_cache = [X]
        prev_outs = X
        for W, b, activation_str in self.layers: # After each iteration, z_cache has a new item and prev_outs has been updated
            next_zs = np.matmul(prev_outs, W) + b
            self.z_cache.append(next_zs)
            activation_fn = self.activation_fns[activation_str]
            prev_outs = activation_fn(self.z_cache[-1])

        return prev_outs[0]


    def backprop(self, errors, learning_rate):
        # There is one more entry in z_cache than in layers. This will allow us
        # to take a pair of z_cache values (neuron levels before activation applied)
        # for each set of weights we want to update. (e.i. one for the units above
        # the connections/weights and one for the units below)
        error_above = errors
        for l in range(len(self.layers)-1, 0, -1):
            z_below, z_above = self.z_cache[l:l+2]

            # If z_below is the network input, don't apply an activation function
            if l==0:
                activation_fn_str = 'identity'
            else:
                activation_fn_str = self.layers[l-1][2]

            activation_fn = self.activation_fns[activation_fn_str]
            act_fn_prime = self.activation_fns[activation_fn_str+'_prime']

            activation_below = activation_fn( z_below )
            trans_act_below = np.reshape(activation_below, newshape=[-1,1])

            error_deriv_prods = act_fn_prime(z_above) * -error_above
            error_deriv_prods = np.reshape(error_deriv_prods, newshape=[1,-1]) # Make a row vector
            #print(z_above, z_below)
            dW = np.matmul(trans_act_below, error_deriv_prods)
            db = error_deriv_prods # End of a lot of math
            #print('error_above={}'.format(error_above))
            #print('db={}'.format(db))

            # Propagate errors to lower layers by reseting error_above
            W, b, _ = self.layers[l]
            error_above = np.matmul(error_above, np.transpose(W))

            # Update weights and biases
            #print('db={}'.format(db))
            #print('W={}'.format(W))
            W += dW*learning_rate
            b += db*learning_rate






if __name__ == '__main__':
    words = []
    num_bits = 3
    for n in range(2**num_bits):
        bits=[]
        for i in range(num_bits):
            bits.append( n % 2 )
            n //= 2
        words.append(bits[::-1])
    words = np.array(words)

    # words now contains list of binary numbers 0 through num_bits**2-1 as list of 1s and 0s
    shifted_words = np.concatenate((words[1:], words[:1]), axis=0)

    # Training samples for performing an incrament operation
    data = zip(words, shifted_words)
    data = list(data)

    nn = NeuralNetwork(layer_info=[(5, 'sig'),
                                   (5, 'sig'),
                                   (3, 'sig')],
                       input_size=3)

    # Train for 10000 epochs (complete iterations through the data)
    start_log_rate = -1
    end_log_rate = -1
    epochs_to_train = 100000
    for epoch in range(epochs_to_train):
        epoch_samples = random.sample(data, len(data))
        for s, (X, Y_) in enumerate(epoch_samples):
            Y = nn.infer(X)
            # e = d(loss)/dY = -(Y_ - Y) = ...
            e = Y - Y_

            # Compute L2 loss derivatives for back prop
            if s==0 and epoch % 10000 == 0:
                loss = sum(0.5*(Y_ - Y)**2)
                print('=========Epoch: {}========='.format(epoch))
                print('loss={}'.format(loss))
                print('{}->{} : {}'.format(X, Y, Y_))
                print('error={}\n'.format(e))

            a = epoch/epochs_to_train
            log_training_rate = start_log_rate*(1-a) + end_log_rate*a
            nn.backprop(e, 10**log_training_rate)

    Y = nn.infer(X)
