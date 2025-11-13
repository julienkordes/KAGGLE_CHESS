from .ppo_trainer import run_ppo_penalty

trainer_dict = {
    "trainer_ppo": run_ppo_penalty,
}

def get_trainer(name: str):
    name = name.lower()
    if name not in trainer_dict:
        raise ValueError(f"Trainer '{name}' not found. Available: {list(trainer_dict.keys())}")
    return trainer_dict[name]
