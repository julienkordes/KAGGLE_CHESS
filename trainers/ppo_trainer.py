import torch

from bbrl.utils.functional import gae
from bbrl_utils.nn import copy_parameters
from bbrl.agents import TemporalAgent
from bbrl_utils.algorithms import iter_partial_episodes
from models.ppo import PPOPenalty
from utils import save_checkpoint


def run_ppo_penalty(ppo: PPOPenalty, args):
    cfg = ppo.cfg

    # The old_policy params must be wrapped into a TemporalAgent
    t_old_policy = TemporalAgent(ppo.old_policy)

    # Training loop
    for train_workspace in ppo.iter_partial_episodes():
        # Run the current policy and evaluate the proba of its action according
        # to the old policy The old_policy can be run after the train_agent on
        # the same workspace because it writes a logprob_predict and not an
        # action. That is, it does not determine the action of the old_policy,
        # it just determines the proba of the action of the current policy given
        # its own probabilities

        # Compute the critic value over the whole workspace
        ppo.t_all_critics(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)
        ws_terminated, ws_reward, ws_v_value, ws_old_v_value = train_workspace[
            "env/terminated",
            "env/reward",
            "critic/v_values",
            "old_critic/v_values",
        ]

        # --- Critic optimization

        # Avoids to extreme V-values (helps stability)
        if cfg.algorithm.clip_range_vf > 0:
            # Clip the difference between old and new values
            # NOTE: this depends on the reward scaling
            ws_v_value = ws_old_v_value + torch.clamp(
                ws_v_value - ws_old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )

        # Compute the advantage using the (clamped) critic values
        with torch.no_grad():
            advantage = gae(
                ws_reward[1:],
                ws_v_value[1:],
                ~ws_terminated[1:],
                ws_v_value[:-1],
                cfg.algorithm.discount_factor,
                cfg.algorithm.gae,
            )
        
        # Compute the critic loss with TD(0)
        target = ws_reward[1:] + cfg.algorithm.discount_factor * ws_old_v_value[1:].detach() * (1 - ws_terminated[1:].int())
        critic_loss = torch.nn.functional.mse_loss(ws_v_value[:-1], target) * cfg.algorithm.critic_coef
        ppo.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ppo.critic_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        ppo.critic_optimizer.step()
        
        # --- Policy optimization

        # We store the advantage into the train_workspace
        if cfg.algorithm.normalize_advantage and advantage.shape[1] > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        train_workspace.set_full("advantage", torch.cat(
            (advantage, torch.zeros(1, advantage.shape[1]))
        ))

        with torch.no_grad():
            # Just computes the probability of the old policy's action
            # to get the ratio of probabilities
            t_old_policy(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                predict_proba=True,
                compute_entropy=False,
            )


        transition_workspace = train_workspace.get_transitions()
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.batch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(
                    cfg.algorithm.batch_size
                )
            else:
                sample_workspace = transition_workspace

            # Compute the policy loss
            
            # Compute the KL divergence
            ppo.t_kl_agent(sample_workspace, t=0, n_steps = 1)
            kl = sample_workspace["kl"][0]
            # Compute the probability of the played actions according to the current policy
            ppo.train_policy(sample_workspace, t=0, n_steps = 1, predict_proba = True, compute_entropy = True)
            # We do not replay the action: we use the one stored into the dataset
            # Note that the policy is not wrapped into a TemporalAgent, but we use a single step
            #Compute the ratio of action probabilities
            Ratio = (sample_workspace["current_policy/logprob_predict"] - sample_workspace["old_policy/logprob_predict"][0]).exp().squeeze(0)
            # Compute the policy loss
            policy_advantage = sample_workspace["advantage"][0]

            policy_loss = (Ratio * policy_advantage - cfg.algorithm.beta  * kl).mean()

            loss_policy = -cfg.algorithm.policy_coef * policy_loss

            # Entropy loss favors exploration
            # Note that the standard PPO algorithms do not have an entropy term, they don't need it
            # because the KL term is supposed to deal with exploration
            # So, to run the standard PPO algorithm, you should set cfg.algorithm.entropy_coef=0
            entropy = sample_workspace["current_policy/entropy"]
            assert len(entropy) == 1, f"{entropy.shape}"
            entropy_loss = entropy[0].mean()
            loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

            # Store the losses for tensorboard display
            ppo.logger.log_losses(critic_loss, entropy_loss, policy_loss, ppo.nb_steps)
            ppo.logger.add_log("advantage", policy_advantage.mean(), ppo.nb_steps)

            ppo.policy_optimizer.zero_grad()
            loss = loss_policy + loss_entropy
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ppo.train_policy.parameters(), cfg.algorithm.max_grad_norm
            )
            ppo.policy_optimizer.step()

        # Copy parameters
        copy_parameters(ppo.train_policy, ppo.old_policy)
        copy_parameters(ppo.critic_agent, ppo.old_critic_agent)

        # Save checkpoint
        save_checkpoint(ppo, ppo.nb_steps, checkpoint_dir=args.save_path+"/checkpoints")
        
        # Evaluate
        ppo.evaluate()