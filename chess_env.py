import torch

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from Chessnut import Game
from utils import material_reward, mobility_reward, win_reward, center_control_reward, promotion_reward, danger_reward, get_legal_moves, check_promotion, fen_to_obs

class ChessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = Game()  
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(14, 8, 8), 
            dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([64, 64])
    
    def reset(self, **kwargs):
        self.game = Game()
        info = {'number': self.game.status}
        return {'obs' : self.get_observation(), 'legal_moves' : get_legal_moves(self.game.get_fen())}, info

    def step(self, action):
        action = Game.i2xy(action[0]) + Game.i2xy(action[1])
        action = check_promotion(action, self.game.board.get_piece(Game.xy2i(action[:2])))
        reward = self.compute_reward(action)
        self.game.apply_move(action)

        unsufficient_materiel = self.unsufficient_materiel()
        length_game = len(self.game.move_history)
        done = (self.game.status == 2) or (self.game.status == 3) or unsufficient_materiel or (length_game > 100)

        legal_moves = -np.ones((218, 2), dtype=np.int64)  # 218 = max legal moves
        if not done:
            legal_moves = get_legal_moves(self.game.get_fen())
        info = {'number': self.game.status}
        return {'obs' : self.get_observation(), 'legal_moves' : legal_moves}, reward, False, done, info

    def unsufficient_materiel(self):
        if self.game.board.find_piece('p') != -1 or self.game.board.find_piece('P') != -1 or self.game.board.find_piece('q') != -1 \
        or self.game.board.find_piece('Q') != -1 or self.game.board.find_piece('r') != -1 or self.game.board.find_piece('R') != -1:
            return False
        if (self.game.board.find_piece('n') != -1 and self.game.board.find_piece('b') != -1) \
        or (self.game.board.find_piece('N') != -1 and self.game.board.find_piece('B') != -1):
            return False
        return True

    def get_observation(self):
        fen = self.game.get_fen()
        return fen_to_obs(fen)

    
    def compute_reward(self, action):
        fen = self.game.get_fen()
        r_mat = material_reward(action, fen)
        r_mob = mobility_reward(action, fen)
        r_win = win_reward(action, fen)
        r_center = center_control_reward(action, fen)
        r_promotion = promotion_reward(action, fen)
        r_danger = danger_reward(action, fen)
        return 0.05 * r_mat + 0.02 * r_center + 0.1 * r_danger + 0.05 * r_mob + r_win + r_promotion 
        