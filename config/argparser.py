import configargparse
from fractions import Fraction


def save_opts(args, fn):
    with open(fn, "w") as fw:
        for items in vars(args):
            fw.write("%s %s\n" % (items, vars(args)[items]))


def str2bool(v):
    v = v.lower()
    if v in ("yes", "true", "t", "1"):
        return True
    elif v in ("no", "false", "f", "0"):
        return False
    raise ValueError(
        "Boolean argument needs to be true or false. " "Instead, it is %s." % v
    )


def fraction_to_float(value):
    try:
        # Try to parse the input as a fraction
        fraction_value = Fraction(value)
        # Convert the fraction to a float
        float_value = float(fraction_value)
        return float_value
    except ValueError:
        # If parsing fails, raise an error
        raise configargparse.argparse.ArgumentTypeError(
            f"{value} is not a valid fraction."
        )

def load_opts():
    get_parser()
    return get_parser().parse_args()

def namespace_to_nested_dict(namespace):
    nested = {}
    for key, value in vars(namespace).items():
        parts = key.split(".")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return nested

def get_parser():
    parser = configargparse.ArgumentParser(description="main")
    parser.register("type", bool, str2bool)
    parser.add_argument(
        "--save_path", type=str, default="experiments/test", help="Path to save the results"
    )
    parser.add_argument(
        "--trainer", type=str, default="trainer_ppo", help="The trainer to use"
    )
    parser.add_argument(
        "--model", type=str, default="PPO", help="The model to use"
    )

    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--save_best", type=bool, default=False)

    # === LOGGER ===
    parser.add_argument("--logger.classname", type=str, default="bbrl.utils.logger.TFLogger")
    parser.add_argument("--logger.cache_size", type=int, default=10000)
    parser.add_argument("--logger.every_n_seconds", type=int, default=10)
    parser.add_argument("--logger.verbose", type=bool, default=False)

    # === ALGORITHM ===
    parser.add_argument("--algorithm.seed", type=int, default=12)
    parser.add_argument("--algorithm.max_grad_norm", type=float, default=0.5)
    parser.add_argument("--algorithm.n_envs", type=int, default=3)
    parser.add_argument("--algorithm.n_steps", type=int, default=30)
    parser.add_argument("--algorithm.eval_interval", type=int, default=1000)
    parser.add_argument("--algorithm.nb_evals", type=int, default=10)
    parser.add_argument("--algorithm.gae", type=float, default=0.8)
    parser.add_argument("--algorithm.discount_factor", type=float, default=0.98)
    parser.add_argument("--algorithm.normalize_advantage", type=bool, default=False)
    parser.add_argument("--algorithm.max_epochs", type=int, default=5000)
    parser.add_argument("--algorithm.opt_epochs", type=int, default=10)
    parser.add_argument("--algorithm.batch_size", type=int, default=256)
    parser.add_argument("--algorithm.clip_range", type=float, default=0.2)
    parser.add_argument("--algorithm.clip_range_vf", type=float, default=0.0)
    parser.add_argument("--algorithm.entropy_coef", type=float, default=2e-7)
    parser.add_argument("--algorithm.policy_coef", type=float, default=1.0)
    parser.add_argument("--algorithm.beta", type=float, default=5.0)
    parser.add_argument("--algorithm.critic_coef", type=float, default=1.0)
    parser.add_argument("--algorithm.policy_type", type=str, default="DiscretePolicy")
    parser.add_argument("--algorithm.architecture.actor_hidden_size", type=int, nargs="+", default=32)
    parser.add_argument("--algorithm.architecture.critic_hidden_size", type=int, nargs="+", default=32)

    # === ENVIRONMENT ===
    parser.add_argument("--gym_env.env_name", type=str, default="ChessEnv-v0")

    # === OPTIMIZER ===
    parser.add_argument("--optimizer.classname", type=str, default="torch.optim.Adam")
    parser.add_argument("--optimizer.lr", type=float, default=1e-3)
    parser.add_argument("--optimizer.eps", type=float, default=1e-5)

    return parser