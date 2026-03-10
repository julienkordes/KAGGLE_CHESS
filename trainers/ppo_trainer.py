import torch
import time

from bbrl.utils.functional import gae
from bbrl_utils.nn import copy_parameters
from bbrl.agents import TemporalAgent
from models.ppo import PPOPenalty
from utils import save_checkpoint


def run_ppo_penalty(ppo: PPOPenalty, args):

    cfg = ppo.cfg
    t_old_policy = TemporalAgent(ppo.old_policy)

    last_checkpoint_time = time.time()
    checkpoint_interval = args.checkpoint_interval * 60

    # -------------------------
    # MAIN TRAINING LOOP
    # -------------------------
    for train_workspace in ppo.iter_partial_episodes():
        
        # ---- 1) Critic evaluation ----
        ppo.t_all_critics(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        ws_terminated, ws_reward, ws_v_value, ws_old_v_value = train_workspace[
            "env/terminated",
            "env/reward",
            "critic/v_values",
            "old_critic/v_values",
        ]

        # Optional V-clipping (helps stability)
        if cfg.algorithm.clip_range_vf > 0:
            ws_v_value = ws_old_v_value + torch.clamp(
                ws_v_value - ws_old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )
        # ---- 2) Compute advantages ----
        with torch.no_grad():
            advantage = gae(
                ws_reward[1:],
                ws_v_value[1:],
                ~ws_terminated[1:],
                ws_v_value[:-1],
                cfg.algorithm.discount_factor,
                cfg.algorithm.gae,
            )

        # TD(0) critic target
        with torch.no_grad():
            target = ws_reward[1:] + cfg.algorithm.discount_factor * ws_old_v_value[1:] * (1 - ws_terminated[1:].int())

        critic_loss = torch.nn.functional.mse_loss(ws_v_value[:-1], target)
        critic_loss = critic_loss * cfg.algorithm.critic_coef

        ppo.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(ppo.critic_agent.parameters(), cfg.algorithm.max_grad_norm)
        ppo.critic_optimizer.step()

        # ---- 3) Store advantage in workspace ----
        if cfg.algorithm.normalize_advantage and advantage.shape[1] > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        train_workspace.set_full(
            "advantage",
            torch.cat((advantage, torch.zeros(1, advantage.shape[1])))
        )

        # ---- 4) Evaluate old policy probabilities ----
        with torch.no_grad():
            t_old_policy(train_workspace, t=0, n_steps=cfg.algorithm.n_steps, predict_proba=True, compute_entropy=False)

        transition_workspace = train_workspace.get_transitions()

        # ===============================================
        #          PPO OPTIMIZATION EPOCHS
        # ===============================================
        for opt_epoch in range(cfg.algorithm.opt_epochs):

            if cfg.algorithm.batch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(cfg.algorithm.batch_size)
            else:
                sample_workspace = transition_workspace

            # Compute KL (monitoring only)
            ppo.t_kl_agent(sample_workspace, t=0, n_steps=1)
            kl = sample_workspace["kl"][0].mean()

            # ---- 5) Current policy forward ----
            ppo.train_policy(sample_workspace, t=0, n_steps=1, predict_proba=True, compute_entropy=True)

            # Ratio = new_prob / old_prob
            log_ratio = sample_workspace["current_policy/logprob_predict"] - sample_workspace["old_policy/logprob_predict"][0]
            ratio = log_ratio.exp().squeeze(0)

            A = sample_workspace["advantage"][0]

            # ---- 6) PPO clipped loss ----
            eps = cfg.algorithm.clip_range

            clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
            surr1 = ratio * A
            surr2 = clipped_ratio * A

            policy_loss = -torch.min(surr1, surr2).mean()

            # ---- 7) Entropy bonus ----
            entropy_loss = -cfg.algorithm.entropy_coef * sample_workspace["current_policy/entropy"][0].mean()

            # ---- 8) Total loss ----
            loss = cfg.algorithm.policy_coef * policy_loss + entropy_loss

            # Backprop
            ppo.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.train_policy.parameters(), cfg.algorithm.max_grad_norm)
            ppo.policy_optimizer.step()


        # ---- 9) Copy params to old policy ----
        copy_parameters(ppo.train_policy, ppo.old_policy)
        copy_parameters(ppo.critic_agent, ppo.old_critic_agent)

        # ---- 10) Checkpoint ----
        current_time = time.time()
        if current_time - last_checkpoint_time > checkpoint_interval:
            save_checkpoint(ppo, ppo.nb_steps, checkpoint_dir=args.save_path + "/checkpoints")
            last_checkpoint_time = current_time

        # ---- 11) Evaluation ----
        ppo.evaluate()
