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
    current_legal_moves = the_board.legal_moves(the_board.opponent(color))
    debugPrint(f'[MIN] Current legal moves: { current_legal_moves }')
 
    if len(current_legal_moves) == 0:
        debugPrint('[MIN] Stopping because found no further possible moves')
        return utility(the_board, color), INVALID_MOVE

    best_move = current_legal_moves[0]
    best_score = -INFINITY

    if time.time() - start_time >= MAX_RUN_TIME:
        debugPrint('[MIN] Stopping because time is up')
        return utility(the_board, color), best_move

    for s in current_legal_moves:
        other_board = board.from_string(str(the_board))
        other_board.process_move(s, the_board.opponent(color))
        
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
    # Score Ratio: Tha ratio of points between our score and the opponent's
    current_score = sum([1 for char in str(the_board) if char == color])
    opponent_score = sum([1 for char in str(the_board) if char == the_board.opponent(color)])
    score_ratio = current_score / opponent_score if opponent_score else current_score

    board_as_string = str(the_board).replace('\n','')
    # Corner Wheight
    current_upper_border_tiles = sum([1 for char in board_as_string[:8] if char == color])
    current_lower_border_tiles = sum([1 for char in board_as_string[-8:] if char == color])
    current_left_border_tiles = sum([1 for char in board_as_string[::8] if char == color])
    current_right_border_tiles = sum([1 for char in board_as_string[7::8] if char == color])
    current_total_border_tiles = current_upper_border_tiles + current_lower_border_tiles + current_left_border_tiles + current_right_border_tiles

    opponent_upper_border_tiles = sum([1 for char in board_as_string[:8] if char == the_board.opponent(color)])
    opponent_lower_border_tiles = sum([1 for char in board_as_string[-8:] if char == the_board.opponent(color)])
    opponent_left_border_tiles = sum([1 for char in board_as_string[::8] if char == the_board.opponent(color)])
    opponent_right_border_tiles = sum([1 for char in board_as_string[7::8] if char == the_board.opponent(color)])
    opponent_total_border_tiles = opponent_upper_border_tiles + opponent_lower_border_tiles + opponent_left_border_tiles + opponent_right_border_tiles

    border_ratio = current_total_border_tiles / opponent_total_border_tiles if opponent_total_border_tiles else current_total_border_tiles
    
    return score_ratio + border_ratio

if __name__ == '__main__':
    b = board.from_file(sys.argv[1])
    f = open('move.txt', 'w')
    f.write('%d,%d' % make_move(b, sys.argv[2]))
    f.close()
