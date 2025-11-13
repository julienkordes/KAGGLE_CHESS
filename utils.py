import os
import torch

import numpy as np

from Chessnut import Game

def material_reward(move, fen):
    game = Game(fen)
    piece_values = {
        'q': 9, 'r': 5, 'n': 3, 'b': 3, 'p': 1,
        'Q': -9, 'R': -5, 'N': -3, 'B': -3, 'P': -1
    }
    to_square = Game.xy2i(move[2:4])
    captured_piece = game.board.get_piece(to_square)
    player = game.state.player
    if player == 'w':
        return piece_values.get(captured_piece, 0)
    else:
        return -piece_values.get(captured_piece, 0)

def mobility_reward(move, fen):
    game = Game(fen)
    player = game.state.player
    game.apply_move(move)
    white_moves = len(list(game.get_moves(player='w')))
    black_moves = len(list(game.get_moves(player='b')))
    reward = (white_moves - black_moves) / (white_moves + black_moves + 1e-6)
    if player == 'w':
        return reward
    else:
        return -reward

def win_reward(move, fen):
    g = Game(fen)
    g.apply_move(move)
    if g.status == Game.CHECKMATE:
        return 100
    return 0

def get_legal_moves(fen, max_moves=218):
    """
    Renvoie un tableau (max_moves, 2) des coups légaux.
    Les coups valides sont encodés comme (from_idx, to_idx).
    Les emplacements inutilisés sont remplis avec (-1, -1).
    """
    game = Game(fen)
    list_moves = game.get_moves()  # liste des coups en UCI

    legal_moves_idx = []
    for uci in list_moves:
        from_idx = Game.xy2i(uci[:2])
        to_idx   = Game.xy2i(uci[2:])
        legal_moves_idx.append((from_idx, to_idx))

    legal_moves_idx = np.array(legal_moves_idx, dtype=np.int64)

    # Padding avec (-1, -1) pour atteindre max_moves
    if legal_moves_idx.shape[0] < max_moves:
        pad_size = max_moves - legal_moves_idx.shape[0]
        padding = -1 * np.ones((pad_size, 2), dtype=np.int64)
        legal_moves_idx = np.vstack([legal_moves_idx, padding])
    elif legal_moves_idx.shape[0] > max_moves:
        legal_moves_idx = legal_moves_idx[:max_moves]  # tronquer si trop grand

    return legal_moves_idx

def check_promotion(move, piece):
    if piece == 'P':
        if move[1] == '7' and move[3] == '8':
            return move[:4] + 'q'
    elif piece == 'p':
        if move[1] == '2' and move[3] == '1':
            return move[:4] + 'q'
    return move

def save_checkpoint(ppo, step, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ppo_checkpoint_{step}.pt")
    
    torch.save({
        "step": step,
        "train_policy_state_dict": ppo.train_policy.state_dict(),
        "old_policy_state_dict": ppo.old_policy.state_dict(),
        "critic_state_dict": ppo.critic_agent.state_dict(),
        "old_critic_state_dict": ppo.old_critic_agent.state_dict(),
        "policy_optimizer_state_dict": ppo.policy_optimizer.state_dict(),
        "critic_optimizer_state_dict": ppo.critic_optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"Checkpoint saved at step {step} → {checkpoint_path}")
