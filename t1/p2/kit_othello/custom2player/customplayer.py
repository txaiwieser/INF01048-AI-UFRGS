import random
import sys
sys.path.append('..')
from common import board
import time

DEBUG = True
MAX_RUN_TIME = 4.0
INVALID_MOVE = (-1, -1)
INFINITY = float('inf')
MAX_DEPTH = 10

def debugPrint(str):
    if DEBUG: print("DEBUG B: " + str)

def make_move(the_board, color):
    """
    Returns the best move from the list of possible ones according to up to MAX_RUN_TIME seconds of execution of minimax with alpha-beta pruning 
    :return: (int, int)
    """
    color = board.Board.WHITE if color == 'white' else board.Board.BLACK

    return decide(the_board, color)

def decide(the_board, color):
    v, m = max_value(the_board, color, INFINITY, -INFINITY, time.time(), MAX_DEPTH)
    debugPrint(f'Found best move: { v }, { m }')
    return m

def max_value(the_board, color, alpha, beta, start_time, remaining_depth):
    current_legal_moves = the_board.legal_moves(color)
    debugPrint(f'[MAX] Current legal moves: { current_legal_moves }')

    if len(current_legal_moves) == 0:
        debugPrint('[MAX] Stopping because found no further possible moves')
        return utility(the_board, color), INVALID_MOVE
    
    best_move = current_legal_moves[0]
    best_score = -INFINITY
    
    if remaining_depth == 0 or time.time() - start_time >= MAX_RUN_TIME:
        debugPrint('[MAX] Stopping because time is up')
        return utility(the_board, color), best_move

    for s in current_legal_moves:
        other_board = board.from_string(str(the_board))
        other_board.process_move(s, color)
        
        min_val = min_value(other_board, the_board.opponent(color), alpha, beta, start_time, remaining_depth - 1)[0]
        if min_val > best_score:
            debugPrint(f'[MAX] Found better move { s }, with utility { min_val }')
            best_score = min_val
            best_move = s
        
        if best_score >= beta:
            debugPrint('[MAX] Alpha-beta pruned')
            return best_score, best_move
        
        alpha = max(alpha, best_score)
    return best_score, best_move

def min_value(the_board, color, alpha, beta, start_time, remaining_depth):
    current_legal_moves = the_board.legal_moves(color)
    debugPrint(f'[MIN] Current legal moves: { current_legal_moves }')
 
    if len(current_legal_moves) == 0:
        debugPrint('[MIN] Stopping because found no further possible moves')
        return utility(the_board, the_board.opponent(color)), INVALID_MOVE

    best_move = current_legal_moves[0]
    best_score = INFINITY

    if remaining_depth == 0 or time.time() - start_time >= MAX_RUN_TIME:
        debugPrint('[MIN] Stopping because time is up')
        return utility(the_board, the_board.opponent(color)), best_move

    for s in current_legal_moves:
        other_board = board.from_string(str(the_board))
        other_board.process_move(s, color)
        
        max_val = max_value(other_board, the_board.opponent(color), alpha, beta, start_time, remaining_depth - 1)[0]
        if max_val < best_score:
            debugPrint(f'[MIN] Found better move { s } with utility { max_val }')
            best_score = max_val
            best_move = s
        
        if best_score <= alpha:
            debugPrint('[MIN] Alpha-beta pruned')
            return best_score, best_move
        
        beta = min(beta, best_score)
    return best_score, best_move

def utility(the_board, color):
    opponent_color = the_board.opponent(color)

    # Board Score: Tha score of points. Number of pieces of our color minus the opponent's pieces
    board_as_string = str(the_board).replace('\n','')
    current_score = sum([1 for char in board_as_string if char == color])
    opponent_score = sum([1 for char in board_as_string if char == opponent_color])
    board_score = current_score - opponent_score
    
    # Corner Weights
    board_weights = [
        +4, -3, +2, +2, +2, +2, -3, +4,
        -3, -4, -1, -1, -1, -1, -4, -3,
        +2, -1, +1, +0, +0, +1, -1, +2,
        +2, -1, +0, +1, +1, +0, -1, +2,
        +2, -1, +0, +1, +1, +0, -1, +2,
        +2, -1, +1, +0, +0, +1, -1, +2,
        -3, -4, -1, -1, -1, -1, -4, -3,
        +4, -3, +2, +2, +2, +2, -3, +4
    ]

    positions_weight = 0

    for index, weight in enumerate(board_weights):
        board_position = board_as_string[index]
        if board_position == color:
            positions_weight += weight
        elif board_position == opponent_color:
            positions_weight -= weight
    
    return 2 * positions_weight + 3 * board_score

if __name__ == '__main__':
    b = board.from_file(sys.argv[1])
    f = open('move.txt', 'w')
    f.write('%d,%d' % make_move(b, sys.argv[2]))
    f.close()
