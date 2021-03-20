import random
import sys
sys.path.append('..')
from common import board
import time

MAX_RUN_TIME = 4.0
INVALID_MOVE = (-1, -1)

def make_move(the_board, color):
    """
    Returns the best move from the list of possible ones according to up to MAX_RUN_TIME seconds of execution of minimax with alpha-beta pruning 
    :return: (int, int)
    """
    color = board.Board.WHITE if color == 'white' else board.Board.BLACK

    return decide(the_board, color)

def decide(the_board, color):
    v, m = max_value(the_board, color, float("-inf"), float("inf"), time.time())
    print('Found best move: ', v, m)
    return m

def max_value(the_board, color, alpha, beta, start_time):
    current_legal_moves = the_board.legal_moves(color)
    print('[MAX] Current legal moves: ', current_legal_moves)

    if len(current_legal_moves) == 0:
        print('[MAX] Stopping because found no further possible moves')
        return utility(the_board, color), INVALID_MOVE
    if time.time() - start_time >= MAX_RUN_TIME:
        print('[MAX] Stopping because time is up')
        return utility(the_board, color), INVALID_MOVE

    best_move = INVALID_MOVE
    for s in current_legal_moves:
        other_board = board.from_string(str(the_board)) # copy board
        other_board.process_move(s, color)
        min_val = min_value(other_board, color, alpha, beta, start_time)[0]
        if alpha < min_val:
            print('[MAX] Found better move ', s, ' with utility ', min_val)
            alpha = min_val
            best_move = s
            if beta > alpha:
                print('[MAX] Aplha-beta pruned')
                return alpha, best_move
    return alpha, best_move

def min_value(the_board, color, alpha, beta, start_time):
    current_legal_moves = the_board.legal_moves(the_board.opponent(color))
    print('[MIN] Current legal moves: ', current_legal_moves)
 
    if len(current_legal_moves) == 0:
        print('[MIN] Stopping because found no further possible moves')
        return utility(the_board, color), INVALID_MOVE
    if time.time() - start_time >= MAX_RUN_TIME:
        print('[MIN] Stopping because time is up')
        return utility(the_board, color), INVALID_MOVE

    best_move = INVALID_MOVE
    for s in current_legal_moves:
        other_board = board.from_string(str(the_board)) # copy board
        other_board.process_move(s, the_board.opponent(color))
        max_val = max_value(other_board, color, alpha, beta, start_time)[0]
        if beta > max_val:
            print('[MIN] Found better move ', s, ' with utility ', max_val)
            beta = max_val
            best_move = s
            if alpha > beta:
                print('[MIN] Aplha-beta pruned')
                return beta, best_move
    return beta, best_move

def utility(the_board, color):
    current_score = sum([1 for char in str(the_board) if char == color])
    opponent_score = sum([1 for char in str(the_board) if char == the_board.opponent(color)])
    
    current_upper_border_tiles = sum([1 for char in str(the_board).replace('\n','')[:8] if char == color])
    current_lower_border_tiles = sum([1 for char in str(the_board).replace('\n','')[-8:] if char == color])
    current_left_border_tiles = sum([1 for char in str(the_board).replace('\n','')[::8] if char == color])
    current_right_border_tiles = sum([1 for char in str(the_board).replace('\n','')[7::8] if char == color])
    current_total_border_tiles = current_upper_border_tiles + current_lower_border_tiles + current_left_border_tiles + current_right_border_tiles

    opponent_upper_border_tiles = sum([1 for char in str(the_board).replace('\n','')[:8] if char == the_board.opponent(color)])
    opponent_lower_border_tiles = sum([1 for char in str(the_board).replace('\n','')[-8:] if char == the_board.opponent(color)])
    opponent_left_border_tiles = sum([1 for char in str(the_board).replace('\n','')[::8] if char == the_board.opponent(color)])
    opponent_right_border_tiles = sum([1 for char in str(the_board).replace('\n','')[7::8] if char == the_board.opponent(color)])
    opponent_total_border_tiles = opponent_upper_border_tiles + opponent_lower_border_tiles + opponent_left_border_tiles + opponent_right_border_tiles

    score_ratio = current_score / opponent_score if opponent_score else current_score
    border_ratio = current_total_border_tiles / opponent_total_border_tiles if opponent_total_border_tiles else current_total_border_tiles

    utility = score_ratio + border_ratio
    
    return utility

if __name__ == '__main__':
    b = board.from_file(sys.argv[1])
    f = open('move.txt', 'w')
    f.write('%d,%d' % make_move(b, sys.argv[2]))
    f.close()
