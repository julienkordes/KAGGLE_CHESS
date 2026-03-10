import gym

from utils import  load_ppo_model, get_legal_moves, fen_to_obs, check_promotion
from config.argparser import load_opts, namespace_to_nested_dict
from Chessnut import Game
from omegaconf import OmegaConf
from models import get_model
from gymnasium.envs.registration import register

args = load_opts()

args_dict = namespace_to_nested_dict(args)
if args_dict["gym_env"]["env_name"] not in gym.envs.registry:
    register(
        id=args_dict["gym_env"]["env_name"],
        entry_point="chess_env:ChessEnv",  
    )
cfg = OmegaConf.create(args_dict)
ppo = get_model(args.model, cfg) 
load_ppo_model(ppo, args.checkpoint_path)
    

def chess_bot(obs):
    fen = obs.board
    game = Game(fen)
    obs_ = fen_to_obs(fen).unsqueeze(0)
    probs_from, probs_to = ppo.eval_agent.agent.agents[1].wrapped.chess_bot(obs_)
    probs_from = probs_from.squeeze(0)
    probs_to = probs_to.squeeze(0)

    legal = get_legal_moves(fen)  
    legal_valid = legal[(legal[:, 0] != -1) & (legal[:, 1] != -1)]
    legal_from = legal_valid[:, 0]
    legal_to = legal_valid[:, 1]
    legal_scores = probs_from[legal_from] * probs_to[legal_to]
    legal_scores = legal_scores / legal_scores.sum()  

    idx = legal_scores.argmax().item()
    action_from = legal_valid[idx, 0]
    action_to = legal_valid[idx, 1]
    action = Game.i2xy(action_from) + Game.i2xy(action_to)
    return check_promotion(action, game.board.get_piece(Game.xy2i(action[:2])))