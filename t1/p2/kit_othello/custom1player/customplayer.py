import sys
sys.path.append('..')
from common import board
import time

DEBUG = False
MAX_RUN_TIME = 3.0
INVALID_MOVE = (-1, -1)
INFINITY = float('inf')
MAX_DEPTH = 10

def debugPrint(str):
    if DEBUG: print("DEBUG A: " + str)

def make_move(the_board, color):
    """
    Returns the best move from the list of possible ones according to up to MAX_RUN_TIME seconds of execution of minimax with alpha-beta pruning 
    :return: (int, int)
    """
    color = board.Board.WHITE if color == 'white' else board.Board.BLACK

    return decide(the_board, color)

def decide(the_board, color):
    v, m = max_value(the_board, color, -INFINITY, INFINITY, time.time(), MAX_DEPTH)
    debugPrint(f'Found best move: { v }, { m }')
    return m

def max_value(the_board, color, alpha, beta, start_time, remaining_depth):
    current_legal_moves = the_board.legal_moves(color)
    debugPrint(f'[MAX] Current legal moves: { current_legal_moves }')

    if len(current_legal_moves) == 0:
        debugPrint('[MAX] Stopping because found no further possible moves')
        return utility(the_board, color), INVALID_MOVE
    
    best_move = current_legal_moves[0]
    
    if time.time() - start_time >= MAX_RUN_TIME:
        debugPrint('[MAX] Stopping because time is up')
        return utility(the_board, color), best_move
    
    if remaining_depth == 0:
        debugPrint('[MAX] Stopping because reach MAX_DEPTH')
        return utility(the_board, color), best_move

    for s in current_legal_moves:
        other_board = board.from_string(str(the_board))
        other_board.process_move(s, color)
        
        min_val = min_value(other_board, the_board.opponent(color), alpha, beta, start_time, remaining_depth - 1)[0]
        debugPrint(f'[MAX] MinVal { min_val }')

        if min_val > alpha:
            debugPrint(f'[MAX] Found better move { s }')
            alpha = min_val
            best_move = s

        if alpha >= beta:
            debugPrint('[MAX] Alpha-beta pruned')
            return alpha, best_move
    return alpha, best_move

def min_value(the_board, color, alpha, beta, start_time, remaining_depth):
    opponent_color = the_board.opponent(color)
    current_legal_moves = the_board.legal_moves(color)
    debugPrint(f'[MIN] Current legal moves: { current_legal_moves }')
 
    if len(current_legal_moves) == 0:
        debugPrint('[MIN] Stopping because found no further possible moves')
        return utility(the_board, opponent_color), INVALID_MOVE

    best_move = current_legal_moves[0]

    if time.time() - start_time >= MAX_RUN_TIME:
        debugPrint('[MAX] Stopping because time is up')
        return utility(the_board, opponent_color), best_move
    
    if remaining_depth == 0:
        debugPrint('[MAX] Stopping because reach MAX_DEPTH')
        return utility(the_board, opponent_color), best_move

    for s in current_legal_moves:
        other_board = board.from_string(str(the_board))
        other_board.process_move(s, color)
        
        max_val = max_value(other_board, opponent_color, alpha, beta, start_time, remaining_depth - 1)[0]
        debugPrint(f'[MIN] MaxVal { max_val }')

        if max_val < beta:
            debugPrint(f'[MIN] Found better move { s }')
            beta = max_val
            best_move = s

        if beta <= alpha:
            debugPrint('[MIN] Alpha-beta pruned')
            return beta, best_move
    return beta, best_move

def utility(the_board, color):
    return heuristic1(the_board, color)

def heuristic1(the_board, color):
    opponent_color = the_board.opponent(color)
    board_as_string = str(the_board).replace('\n','')

    # Score Ratio: Tha ratio of points between our score and the opponent's
    current_score = sum([1 for char in board_as_string if char == color])
    opponent_score = sum([1 for char in board_as_string if char == opponent_color])
    score_ratio = current_score / opponent_score if opponent_score else current_score
    
    # Corner Weight
    current_upper_border_tiles = sum([1 for char in board_as_string[:8] if char == color])
    current_lower_border_tiles = sum([1 for char in board_as_string[-8:] if char == color])
    current_left_border_tiles = sum([1 for char in board_as_string[::8] if char == color])
    current_right_border_tiles = sum([1 for char in board_as_string[7::8] if char == color])
    current_total_border_tiles = current_upper_border_tiles + current_lower_border_tiles + current_left_border_tiles + current_right_border_tiles

    opponent_upper_border_tiles = sum([1 for char in board_as_string[:8] if char == opponent_color])
    opponent_lower_border_tiles = sum([1 for char in board_as_string[-8:] if char == opponent_color])
    opponent_left_border_tiles = sum([1 for char in board_as_string[::8] if char == opponent_color])
    opponent_right_border_tiles = sum([1 for char in board_as_string[7::8] if char == opponent_color])
    opponent_total_border_tiles = opponent_upper_border_tiles + opponent_lower_border_tiles + opponent_left_border_tiles + opponent_right_border_tiles

    border_ratio = current_total_border_tiles / opponent_total_border_tiles if opponent_total_border_tiles else current_total_border_tiles
    
    return score_ratio + (2 * border_ratio)

def heuristic2(the_board, color):
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
