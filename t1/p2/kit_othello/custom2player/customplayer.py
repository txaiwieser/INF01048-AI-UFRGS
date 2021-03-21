import random
import sys
sys.path.append('..')
from common import board
import time

DEBUG = False
MAX_RUN_TIME = 4.0
INVALID_MOVE = (-1, -1)
INFINITY = float('inf')

def debugPrint(str):
    if DEBUG: print("DEBUG: " + str)

def make_move(the_board, color):
    """
    Returns the best move from the list of possible ones according to up to MAX_RUN_TIME seconds of execution of minimax with alpha-beta pruning 
    :return: (int, int)
    """
    color = board.Board.WHITE if color == 'white' else board.Board.BLACK

    return decide(the_board, color)

def decide(the_board, color):
    v, m = max_value(the_board, color, INFINITY, -INFINITY, time.time())
    debugPrint(f'Found best move: { v }, { m }')
    return m

def max_value(the_board, color, alpha, beta, start_time):
    current_legal_moves = the_board.legal_moves(color)
    debugPrint(f'[MAX] Current legal moves: { current_legal_moves }')

    if len(current_legal_moves) == 0:
        debugPrint('[MAX] Stopping because found no further possible moves')
        return utility(the_board, color), INVALID_MOVE
    
    best_move = current_legal_moves[0]
    best_score = -INFINITY
    
    if time.time() - start_time >= MAX_RUN_TIME:
        debugPrint('[MAX] Stopping because time is up')
        return utility(the_board, color), best_move

    for s in current_legal_moves:
        other_board = board.from_string(str(the_board))
        other_board.process_move(s, color)
        
        min_val = min_value(other_board, color, alpha, beta, start_time)[0]
        if min_val > best_score:
            debugPrint(f'[MAX] Found better move { s }, with utility { min_val }')
            best_score = min_val
            best_move = s
        
        if best_score >= beta:
            debugPrint('[MAX] Aplha-beta pruned')
            return best_score, best_move
        
        alpha = max(alpha, best_score)
    return best_score, best_move

def min_value(the_board, color, alpha, beta, start_time):
    opponent_color = the_board.opponent(color)
    current_legal_moves = the_board.legal_moves(opponent_color)
    debugPrint(f'[MIN] Current legal moves: { current_legal_moves }')
 
    if len(current_legal_moves) == 0:
        debugPrint('[MIN] Stopping because found no further possible moves')
        return utility(the_board, color), INVALID_MOVE

    best_move = current_legal_moves[0]
    best_score = INFINITY

    if time.time() - start_time >= MAX_RUN_TIME:
        debugPrint('[MIN] Stopping because time is up')
        return utility(the_board, color), best_move

    for s in current_legal_moves:
        other_board = board.from_string(str(the_board))
        other_board.process_move(s, opponent_color)
        
        max_val = max_value(other_board, color, alpha, beta, start_time)[0]
        if max_val < best_score:
            debugPrint(f'[MIN] Found better move { s } with utility { max_val }')
            best_score = max_val
            best_move = s
        
        if best_score <= alpha:
            debugPrint('[MIN] Aplha-beta pruned')
            return best_score, best_move
        
        beta = min(beta, best_score)
    return best_score, best_move

def utility(the_board, color):
    # Board Score: Tha score of points. Number of pieces of our color minus the opponent's pieces
    board_as_string = str(the_board).replace('\n','')
    current_score = sum([1 for char in board_as_string if char == color])
    opponent_score = sum([1 for char in board_as_string if char == the_board.opponent(color)])
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
        if board_as_string[index] == color:
            positions_weight += weight
        else:
            positions_weight -= weight
    
    return positions_weight + board_score

if __name__ == '__main__':
    b = board.from_file(sys.argv[1])
    f = open('move.txt', 'w')
    f.write('%d,%d' % make_move(b, sys.argv[2]))
    f.close()
