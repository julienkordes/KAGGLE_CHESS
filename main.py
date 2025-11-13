from Chessnut import Game
import random

from utils import material_reward, mobility_reward, win_reward

def chess_bot(obs):
    game = Game(obs.board)
    moves = list(game.get_moves())
    best_move = random.choice(moves)
    best_reward = 0
    for move in moves:
        r_mat = material_reward(move, obs.board)
        r_mob = mobility_reward(move, obs.board)
        r_win = win_reward(move, obs.board)
        r = 0.1 * r_mat + 0.01 * r_mob + r_win
        if r > best_reward:
            best_reward = r
            best_move = move
    return best_move
