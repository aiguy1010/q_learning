import numpy as np

BLANK = 0
FOOD = 1
SPIKES = -1
WALL = -10
PLAYER = 10

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

board_size = 25
board = np.zeros(shape=[board_size, board_size])
player_pos = [board_size//2, board_size//2]
board[player_pos[0], player_pos[1]] = PLAYER

def place_tiles(tile_type, prob):
    global baord
    global board_size
    for i in range(board_size):
        for j in range(board_size):
            if board[i,j] != 0:
                continue
            if random.random() <= prob:
                board[i,j]=tile_type

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
        board[player_pos[0], player_pos[1]] = BLANK
        board[new_pos[0], new_pos[1]] = PLAYER
        player_pos = new_pos
        return 0
    elif board[new_pos[0], new_pos[1]] == FOOD:
        board[player_pos[0], player_pos[1]] = BLANK
        board[new_pos[0], new_pos[1]] = PLAYER
        player_pos = new_pos
        return FOOD
    elif board[new_pos[0], new_pos[1]] == WALL:
        return 0
    elif board[new_pos[0], new_pos[1]] == SPIKES:
        return SPIKES
