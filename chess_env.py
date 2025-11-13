import torch

import gymnasium as gym
import numpy as np

from gymnasium import spaces
from Chessnut import Game
from utils import material_reward, mobility_reward, win_reward, get_legal_moves, check_promotion

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
        if self.game.board.find_piece('p') != -1 or self.game.board.find_piece('P') != -1 or self.game.board.find_piece('q') != -1 or self.game.board.find_piece('Q') != -1 or self.game.board.find_piece('r') != -1 or self.game.board.find_piece('R') != -1:
            return False
        if (self.game.board.find_piece('n') != -1 and self.game.board.find_piece('b') != -1) or (self.game.board.find_piece('N') != -1 and self.game.board.find_piece('B') != -1):
            return False
        return True

    def get_observation(self):
        """
        Convertit un FEN en tenseur (14, 8, 8) pour l'observation.
        Canaux :
        0–5  : pièces blanches (P, N, B, R, Q, K)
        6–11 : pièces noires (p, n, b, r, q, k)
        12   : trait (1 si blanc, 0 sinon)
        13   : roques et en-passant
        """
        fen = self.game.get_fen()
        piece_to_channel = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        tensor = np.zeros((14, 8, 8), dtype=np.float32)
        
        parts = fen.split()
        placement, side, castling, en_passant = parts[0:4]

        ranks = placement.split('/')
        for row, rank in enumerate(ranks):
            col = 0
            for char in rank:
                if char.isdigit():
                    col += int(char)
                else:
                    channel = piece_to_channel[char]
                    tensor[channel, row, col] = 1.0
                    col += 1

        tensor[12, :, :] = 1.0 if side == 'w' else 0.0

        if 'K' in castling: tensor[13, 7, 7] = 1.0  # roi blanc
        if 'Q' in castling: tensor[13, 7, 0] = 1.0  # dame blanche
        if 'k' in castling: tensor[13, 0, 7] = 1.0  # roi noir
        if 'q' in castling: tensor[13, 0, 0] = 1.0  # dame noire

        if en_passant != '-':
            file = ord(en_passant[0]) - ord('a')
            rank = 8 - int(en_passant[1])
            tensor[13, rank, file] = 1.0

        tensor = tensor[:, ::-1, :]
        return torch.tensor(tensor.copy())

    
    def compute_reward(self, action):
        fen = self.game.get_fen()
        r_mat = material_reward(action, fen)
        r_mob = mobility_reward(action, fen)
        r_win = win_reward(action, fen)
        return 0.1 * r_mat + 0.01 * r_mob + r_win
        