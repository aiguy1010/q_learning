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
    """Returns a window with shape [1+2*view_margin, 1+2*view_margin]"""
    global player_pos
    global board_size, board
    global WALL

    pi, pj = player_pos
    i_ref, j_ref = pi-view_margin, pj-view_margin
    window = np.zeros(shape=[1+2*view_margin, 1+2*view_margin])
    for i in range(pi-view_margin, pi+view_margin):
        for j in range(pj-view_margin, pj+view_margin):
            if i < 0 or j < 0 or i >= board_size or j >= board_size:
                window[i,j] = WALL
            else:
                window[i-i_ref,j-j_ref] = board[i, j]

    return window

def do_move(move):
    """Updates the board and returns a reward value (can be zero)"""
    global UP, DOWN, LEFT, RIGHT
    global player_pos, board_size

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

def run_world(nn, steps, view_margin, learning_rate=None, display=True, misstep_prop=.0):
    # Initialize the board
    init_board(food_prob=0.1, spikes_prob=0.05)

    for step in range(steps):
        # Get a snapshot of a window around the player and flatten it (have to feed it to FC layer)
        visable_window = read_input_window(view_margin)
        flat_window = visable_window.reshape(shape=[1, -1])

        # Get an input to feed the learner
        view_window = read_input_window(view_margin)

        # Run inference with nn to get a Q value for each action in this state
        action_qvals = nn.infer(view_window.reshape(shape=[1, -1]))
        best_action = action_qvals.index(max(action_qvals))

        # Do misstep?
        if random.random() < misstep:
            possible_actions = [0, 1, 2, 3]
            take_action = random.choice( possible_actions[:best_action] + possible_actions[best_action+1:] )
        else:
            take_action = best_action

        # Do the move/action
        do_move(take_action)

        if learning_rate is not None:
            nn.push



if __name__ == '__main__':
    # Initialize Pygame
    tile_size = 32
    screen_size = screen_width, screen_height = board_size*tile_size, board_size*board_size
    pygame.init()
    display_surf = pygame.display.set_mode(screen_size)

    # Initialize the Neural Network
    view_margin = 2
    nn = NeuralNetwork(layer_info=[(50, 'sig'),
                                   (50, 'sig'),
                                   (16, 'sig'),
                                   (4, 'none')],
                       input_size=2*view_margin+1)

    # Do some training
    run_world(nn, 1000, learning_rate=0.1, display=False, misstep_prop=0.5)
    run_world(nn, 10000, learning_rate=0.1, display=False, misstep_prop=0.1)

    # Initialize the board
    init_board(food_prob=0.1, spikes_prob=0.05)

    while True:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # Get an input to feed the learning
        view_window =

        # Run inference with nn to get a Q value for each action in this state
        action_qvals = nn.infer(view_window.reshape(shape=[1, -1]))

        # Draw everything
        draw_world(display_surf, tile_size)
        pygame.display.flip()
