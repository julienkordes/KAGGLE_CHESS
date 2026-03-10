import os
import torch

import numpy as np

from Chessnut import Game

piece_values = {
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 100
}

def see(game, target_idx, player):
    """
    Static Exchange Evaluation pour la case target_idx.
    Retourne le gain matériel net pour 'player' si on capture ou se fait capturer.
    """
    # Récupérer les attaquants du joueur et de l'adversaire
    def get_attackers(game, target_idx, player):
        """
        Renvoie tous les attaquants d'une case, y compris ceux bloqués par d'autres pièces.
        player: 'w' ou 'b', celui qui considère la capture.
        """
        board = game.board
        attackers = []

        opponent = 'b' if player == 'w' else 'w'

        # Indices de la case cible
        target_rank, target_file = divmod(target_idx, 8)

        # Directions pour les tours et dames (horizontales et verticales)
        rook_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Directions pour les fous et dames (diagonales)
        bishop_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        # Cavaliers
        knight_moves = [(-2,-1), (-1,-2), (-2,1), (-1,2), (1,-2), (2,-1), (1,2), (2,1)]
        # Pions
        pawn_dirs = [(-1,-1), (-1,1)] if player == 'b' else [(1,-1), (1,1)]
        
        def in_board(r, f):
            return 0 <= r < 8 and 0 <= f < 8

        # 1. Tours et dames horizontales/verticales
        for dr, df in rook_dirs:
            r, f = target_rank + dr, target_file + df
            while in_board(r, f):
                idx = r*8 + f
                piece = board.get_piece(idx)
                if piece != ' ':
                    owner = 'w' if piece.isupper() else 'b'
                    if owner == player:
                        # adversaire
                        if piece.lower() in ['r', 'q']:
                            attackers.append((idx, piece))
                        break
                    else:
                        # pièce amie
                        if piece.lower() in ['r', 'q']:
                            # continue la ligne, ne compte pas comme attaquant direct
                            r += dr
                            f += df
                            continue
                        else:
                            # pièce amie bloquante
                            break
                r += dr
                f += df

      # 2. Fous et dames diagonales
        for dr, df in bishop_dirs:
            r, f = target_rank + dr, target_file + df
            steps = 1  # distance depuis la target
            behind_mode = False  # actif si pion ami à distance 1

            while in_board(r, f):
                idx = r*8 + f
                piece = board.get_piece(idx)

                if piece != ' ':
                    owner = 'w' if piece.isupper() else 'b'
                    piece_type = piece.lower()

                    # -----------------------
                    # MODE NORMAL
                    # -----------------------
                    if not behind_mode:

                    # CAS PION AMI À UNE CASE
                    # pion ami à une case et bien orienté ?
                        if owner == player and piece_type == 'p' and steps == 1:
                            # orientation correcte
                            if (player == 'b' and dr == -1) or (player == 'w' and dr == 1):
                                behind_mode = True
                                r += dr
                                f += df
                                steps += 1
                                continue
                            else:
                                break  # pion ami non orienté → stop total

                        # FOUS/DAMES AMIS = attaquants et on s'arrête
                        if owner == player and piece_type in ['b', 'q']:
                            attackers.append((idx, piece))
                        break  # autres pièces → stop total

                    # -----------------------
                    # MODE BEHIND_MODE
                    # -----------------------
                    else:
                        # derrière un pion ami → on cherche fou/dame ami
                        if owner == player and piece_type in ['b', 'q']:
                            attackers.append((idx, piece))
                            # et on continue encore !
                            r += dr
                            f += df
                            steps += 1
                            continue

                        # sinon on arrête après cette case
                        break

                # case vide
                r += dr
                f += df
                steps += 1


        # 3. Cavaliers
        for dr, df in knight_moves:
            r, f = target_rank + dr, target_file + df
            if in_board(r, f):
                idx = r*8 + f
                piece = board.get_piece(idx)
                if piece != ' ' and (('w' if piece.isupper() else 'b') == player) and piece.lower() == 'n':
                    attackers.append((idx, piece))

        # 4. Pions
        for dr, df in pawn_dirs:
            r, f = target_rank + dr, target_file + df
            if in_board(r, f):
                idx = r*8 + f
                piece = board.get_piece(idx)
                if piece != ' ' and (('w' if piece.isupper() else 'b') == player) and piece.lower() == 'p':
                    attackers.append((idx, piece))

        # 5. Roi
        king_moves = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dr, df in king_moves:
            r, f = target_rank + dr, target_file + df
            if in_board(r, f):
                idx = r*8 + f
                piece = board.get_piece(idx)
                if piece != ' ' and (('w' if piece.isupper() else 'b') == player) and piece.lower() == 'k':
                    attackers.append((idx, piece))
        return attackers


    opponent = 'b' if player == 'w' else 'w'
    my_attackers = sorted(get_attackers(game, target_idx, player),
                          key=lambda x: piece_values[x[1].lower()])
    opp_attackers = sorted(get_attackers(game, target_idx, opponent),
                           key=lambda x: piece_values[x[1].lower()])

    target_piece = game.board.get_piece(target_idx)
    target_value = piece_values.get(target_piece.lower(), 0)

    gain = [target_value]
    side = opponent
    my_list = list(my_attackers)
    opp_list = list(opp_attackers)

    while my_list or opp_list:
        if side == player:
            if not my_list: break
            idx, piece = my_list.pop(0)
        else:
            if not opp_list: break
            idx, piece = opp_list.pop(0)
        d = piece_values[piece.lower()] 
        gain.append(d)
        side = 'b' if side == 'w' else 'w'
    somme = 0
    for i in range(len(gain)-1):
        if i % 2 == 0:
            somme -= gain[i]
        else:
            somme += gain[i]
    return somme

def danger_reward(move, fen):
    game = Game(fen)
    player = game.state.player
    # Index de la case où la pièce va
    target_idx = Game.xy2i(move[2:4])
    game.apply_move(move)
    se_val = see(game, target_idx, player)
    if se_val < 0:
        # pénalité proportionnelle au gain négatif
        return se_val
    return 0


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
        return 1
    return 0

def center_control_reward(move, fen):
    to_sq = move[2:4]
    if to_sq in ["d4", "e4", "d5", "e5"]:
        return 1
    return 0

def promotion_reward(move, fen):
    game = Game(fen)
    piece = game.board.get_piece(Game.xy2i(move[:2]))
    if piece == 'P' and move[3] == '8':
        return 1  
    elif piece == 'p' and move[3] == '1':
        return 1  
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

def load_ppo_model(ppo_instance, filename):
    # Chargement du fichier
    checkpoint = torch.load(filename)
    
    # Restauration des états des modèles et des optimiseurs
    ppo_instance.train_policy.load_state_dict(checkpoint['train_policy_state_dict'])
    ppo_instance.critic_agent.load_state_dict(checkpoint['critic_state_dict'])
    ppo_instance.old_policy.load_state_dict(checkpoint['old_policy_state_dict'])
    ppo_instance.old_critic_agent.load_state_dict(checkpoint['old_critic_state_dict'])
    ppo_instance.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    ppo_instance.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    # Restauration de la configuration (si nécessaire)
    ppo_instance.cfg = checkpoint.get('config', ppo_instance.cfg)
    
    print(f"Modèle PPO chargé depuis le fichier : {filename}")

def fen_to_obs(fen):
        """
        Convertit un FEN en tenseur (14, 8, 8) pour l'observation.
        Canaux :
        0–5  : pièces blanches (P, N, B, R, Q, K)
        6–11 : pièces noires (p, n, b, r, q, k)
        12   : trait (1 si blanc, 0 sinon)
        13   : roques et en-passant
        """
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