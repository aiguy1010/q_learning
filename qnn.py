import numpy as np
import random
import sys
import pygame
from pygame.locals import *
from nn import NeuralNetwork

BLANK = 0
FOOD = 1
SPIKES = -1
WALL = -10

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

board_size = 30
board = np.zeros(shape=[board_size, board_size])
player_pos = [board_size//2, board_size//2]

lambda_discount = 0.8

def place_tiles(tile_type, prob):
    global baord
    global board_size
    for i in range(board_size):
        for j in range(board_size):
            if board[i,j] != 0:
                continue
            if random.random() <= prob:
                board[i,j]=tile_type


def init_board(food_prob, spikes_prob):
    global board
    global board_size
    global player_pos
    board = np.zeros(shape=[board_size, board_size])
    player_pos = [board_size//2, board_size//2]
    place_tiles(FOOD, food_prob)
    place_tiles(SPIKES, spikes_prob)


def read_input_window(view_margin):
    """Returns a window with shape (1+2*view_margin, 1+2*view_margin)"""
    global player_pos
    global board_size, board
    global WALL

    pi, pj = player_pos
    i_ref, j_ref = pi-view_margin, pj-view_margin
    window = np.zeros(shape=[1+2*view_margin, 1+2*view_margin])
    for i in range(pi-view_margin, pi+view_margin):
        for j in range(pj-view_margin, pj+view_margin):
            if i < 0 or j < 0 or i >= board_size or j >= board_size:
                window[i-i_ref,j-j_ref] = WALL
            else:
                window[i-i_ref,j-j_ref] = board[i, j]

    return window


def do_move(move):
    """Updates the board and returns a reward value (can be zero)"""
    global UP, DOWN, LEFT, RIGHT
    global player_pos, board, board_size

    # Where are we going
    new_pos = np.array(player_pos)
    if move == UP:
        new_pos[0] = max(player_pos[0]-1, 0)
    elif move == DOWN:
        new_pos[0] = min(player_pos[0]+1, board_size-1)
    elif move == LEFT:
        new_pos[1] = max(player_pos[1]-1, 0)
    elif move == RIGHT:
        new_pos[1] = min(player_pos[1]+1, board_size-1)

    # What happens when we step?
    if board[new_pos[0], new_pos[1]] == BLANK:
        player_pos = new_pos
        return 0
    elif board[new_pos[0], new_pos[1]] == FOOD:
        player_pos = new_pos
        board[new_pos[0], new_pos[1]] = BLANK
        return FOOD
    elif board[new_pos[0], new_pos[1]] == WALL:
        return 0
    elif board[new_pos[0], new_pos[1]] == SPIKES:
        player_pos = new_pos
        return SPIKES


def draw_world(display_surf, tile_size):
    global board, board_size
    global player_pos
    global FOOD, SPIKES, BLANK, WALL

    print('player_pos={}'.format(player_pos))
    player_color = (0, 0, 255)
    food_color = (0, 255, 0)
    spike_color = (255, 0, 0)
    wall_color = (100, 100, 100)
    blank_color = (0, 0, 0)
    for i in range(board_size):
        for j in range(board_size):
            tile_rect = pygame.Rect(j*tile_size, i*tile_size, tile_size, tile_size)
            color_to_draw = blank_color
            if player_pos[0] == i and player_pos[1] == j:
                color_to_draw = player_color
            elif board[i, j] == FOOD:
                color_to_draw = food_color
            elif board[i, j] == SPIKES:
                color_to_draw = spike_color
            elif board[i, j] == WALL:
                color_to_draw = wall_color
            pygame.draw.rect(display_surf, color_to_draw, tile_rect)


def run_world(nn, rounds, steps_per_round, view_margin=5, learning_rate=None, display_surf=None, misstep_prob=.0):
    for r in range(rounds):
        # Initialize the board
        init_board(food_prob=0.1, spikes_prob=0.05)

        for step in range(steps_per_round):
            # Display?
            if display_surf is not None:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: sys.exit()

                # Draw everything
                draw_world(display_surf, tile_size)
                pygame.display.flip()

                # Wait....
                pygame.time.wait(500)

            # Get a snapshot of a window around the player and flatten it (have to feed it to FC layer)
            visable_window = read_input_window(view_margin)
            flat_window = visable_window.reshape([1, -1])

            # Get an input to feed the learner
            view_window = read_input_window(view_margin)

            # Run inference with nn to get a Q value for each action in this state
            action_qvals = nn.infer(view_window.reshape([1, -1]))
            #print('action_qvals={}'.format(action_qvals))
            best_action = np.argmax(action_qvals)

            # Do misstep?
            if random.random() < misstep_prob:
                possible_actions = [0, 1, 2, 3]
                take_action = random.choice( possible_actions[:best_action] + possible_actions[best_action+1:] )
            else:
                take_action = best_action

            # Do the move/action
            step_cost = 0.2
            R = do_move(take_action) - step_cost

            # Update Q-function if learning is enabled
            if learning_rate is not None and len(nn.z_cache_stash)!=0:
                # Get ready to backprop on old activation levels
                nn.pushfront_z_cache()
                nn.popback_z_cache()
                prev_qs = nn.z_cache[-1][0]
                prev_best_action = prev_qs.argmax(prev_qs)
                #print( 'previous_qs={}'.format(previous_qs) )

                # backprop toward Q for this
                target_q = R + lambda_discount*prev_qs[prev_best_action]
                target_qs = np.array(previous_qs) 
                errors = np.zeros(4)
                errors[best_index] = previous_qs[best_index] - target_q
                nn.backprop()
        print('Finished round #{}'.format(r+1))



if __name__ == '__main__':
    # Initialize Pygame
    tile_size = 32
    screen_size = screen_width, screen_height = board_size*tile_size, board_size*board_size
    pygame.init()
    display_surf = pygame.display.set_mode(screen_size)

    # Initialize the Neural Network
    view_margin = 1
    nn = NeuralNetwork(layer_info=[(100, 'sig'),
                                   (100, 'sig'),
                                   (16, 'sig'),
                                   (4, 'identity')],
                       input_size=(2*view_margin+1)**2 )

    # Do some training
    run_world(nn, 100, 100, view_margin, learning_rate=0.5, display_surf=None, misstep_prob=0.5)
    run_world(nn, 100, 100, view_margin, learning_rate=0.1, display_surf=None, misstep_prob=0.3)
    run_world(nn, 1000, 500, view_margin, learning_rate=0.1, display_surf=None, misstep_prob=0.1)

    # Do a display run!
    run_world(nn, 100, 100, view_margin, learning_rate=0.0, display_surf=display_surf, misstep_prob=0.0)
