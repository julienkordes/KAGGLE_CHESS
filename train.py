import os
import gym

from omegaconf import OmegaConf
from config.argparser import load_opts, save_opts, namespace_to_nested_dict
from trainers import get_trainer
from models import get_model
from gymnasium.envs.registration import register

def main():
    args = load_opts()

    exp_dir = args.save_path
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, "config.txt")
    save_opts(args, config_path)

    args_dict = namespace_to_nested_dict(args)
    if args_dict["gym_env"]["env_name"] not in gym.envs.registry:
        register(
            id=args_dict["gym_env"]["env_name"],
            entry_point="chess_env:ChessEnv",  
        )
    cfg = OmegaConf.create(args_dict)
    trainer = get_trainer(args.trainer)
    ppo_kl = get_model(args.model, cfg)
    trainer(ppo_kl, args)

if __name__ == "__main__":
    main()