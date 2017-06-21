import numpy as np
import random

miss_step_prob = 0.1
step_cost = 0.1

width, height = 10, 10

board = np.zeros([width, height])
board[0, width-1] = -1
board[height-1, width-1] = 1

state = (0, 0)

Q = np.zeros([width, height, 4])
gamma = 0.85 # <-- future reward discount rate
alpha = 0.75 # <-- learning rate

def run(steps, train=False, verbose=False):
    global state, board, Q, alpha, gamma, possible_moves
    for i in range(steps):
        # Pick a move
        if train:
            best_move = rnd_pick_best( Q[state[0], state[1], :] )
        else:
            best_move, best_Q = None, None
            for m in range(4):
                Q_m = Q[state[0], state[1], m]
                if best_Q is None or Q_m > best_Q:
                    best_Q = Q_m
                    best_move = m

        # Do step
        old_state = state
        if random.random() < miss_step_prob:
            wrong_moves = [0,1,2,3]
            wrong_moves.remove(best_move)
            update_state(random.choice(wrong_moves))
        else:
            update_state(best_move)

        if train:
            # Update Q value for old state and action
            old_Q = Q[old_state[0], old_state[1], best_move]
            Q[old_state[0], old_state[1], best_move] = old_Q + alpha*(board[state]-step_cost + gamma*max(Q[state[0], state[1], :]) - old_Q)

        if verbose:
            print('{} -> {}'.format(old_state, state))


def update_state(move):
    global state, width, height
    if move == 0: # Up
        new_i = max(0, state[0]-1)
        state = (new_i, state[1])
    elif move == 1: # Down
        new_i = min(height-1, state[0]+1)
        state = (new_i, state[1])
    elif move == 2: # Left
        new_j = max(0, state[1]-1)
        state = (state[0], new_j)
    else:
        new_j = min(width-1, state[1]+1)
        state = (state[0], new_j)

def rnd_pick_best(options):
    """Returns index of picked option"""
    exp_opts = np.exp(options)
    logits = exp_opts / sum(exp_opts)

    boundaries = []
    total = 0
    for l in logits:
        total += l
        boundaries.append(total)

    # Return the random choice
    x = random.random()
    for i in range(len(options)):
        if x <= boundaries[i]:
            return i


if __name__ == '__main__':
    for t in range(100):
        run(1000, train=True)
        state = (0,0)
    print(Q)

    # Run!
    run(100, verbose=True)
