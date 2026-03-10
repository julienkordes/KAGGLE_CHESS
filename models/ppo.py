import torch
import random
import copy

import torch.nn as nn
import numpy as np

from bbrl.agents import Agent, Agents, KWAgentWrapper, TemporalAgent
from bbrl_utils.algorithms import EpisodicAlgo
from bbrl_utils.nn import setup_optimizer


class CNN_pred(nn.Module):
    def __init__(self, state_dim, hidden_size, activation, V_agent = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64*8*8, 256)
        self.V_agent = V_agent
        if V_agent:
            self.head = nn.Linear(256, 1)
        else:
            self.from_head = nn.Linear(256, 64)  # from_idx
            self.to_head = nn.Linear(256, 64)    # to_idx
        self.activation = activation


    def forward(self, obs):
        x = self.conv(obs)            # obs: (B,14,8,8)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc(x))
        if self.V_agent:
            critic = self.head(x)
            return critic
        else:
            from_logits = self.from_head(x)
            to_logits = self.to_head(x)
            return from_logits, to_logits
    

class KLAgent(Agent):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, t, **kwargs):
        obs = self.get(("env/env_obs/obs", t))
        dist_1_from, dist_1_to = self.model_1.dist(obs)
        dist_2_from, dist_2_to = self.model_2.dist(obs)
        kl_from = torch.distributions.kl.kl_divergence(dist_1_from, dist_2_from)
        kl_to = torch.distributions.kl.kl_divergence(dist_1_to, dist_2_to)
        kl_total = kl_from + kl_to
        self.set(("kl", t), kl_total)

class VAgent(Agent):
    def __init__(self, state_dim, hidden_size, name="critic"):
        super().__init__(name=name)
        self.is_q_function = False
        self.model = CNN_pred(
                state_dim, hidden_size, activation=nn.ReLU(), V_agent=True
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs/obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set((f"{self.prefix}v_values", t), critic)

class DiscretePolicy(Agent):
    def __init__(self, state_dim, hidden_size, name="policy"):
        super().__init__(name=name)
        self.model = CNN_pred(
            state_dim, hidden_size, activation=nn.ReLU()
        )

    def dist(self, obs):
        from_logits, to_logits = self.model(obs)
        to_probs = torch.softmax(to_logits, dim=-1)
        from_probs = torch.softmax(from_logits, dim=-1)
        return torch.distributions.Categorical(from_probs), torch.distributions.Categorical(to_probs)

    def forward(
        self,
        t,
        *,
        stochastic=True,
        predict_proba=False,
        compute_entropy=False,
        **kwargs,
    ):
        obs = self.get(("env/env_obs/obs", t))  # (B,14,8,8)
        from_logits, to_logits = self.model(obs)
        from_prob = torch.softmax(from_logits, dim=-1)  # (B,64)
        to_prob = torch.softmax(to_logits, dim=-1)      # (B,64)

        # Récupérer les coups légaux pour chaque élément du batch
        legal_moves_idx = self.get(("env/env_obs/legal_moves", t))  # shape (B, max_moves, 2)

        if predict_proba:
            # Prévoir log_probs des actions déjà prises
            action = self.get(("action", t))  # shape (B, 2)
            log_probs = []
            for b in range(action.size(0)):
                from_idx = action[b, 0]
                to_idx   = action[b, 1]
                log_prob = from_prob[b, from_idx].log() + to_prob[b, to_idx].log()
                log_probs.append(log_prob)
            log_probs = torch.stack(log_probs)
            self.set((f"{self.prefix}logprob_predict", t), log_probs)
        else:
            from_list = []
            to_list = []
            for b in range(obs.size(0)):
                if not self.get(("env/done", t))[b]:
                    legal = legal_moves_idx[b]  # (max_moves, 2)

                    # Filtrer les coups valides (ignore le padding (-1, -1))
                    legal_valid = legal[(legal[:, 0] != -1) & (legal[:, 1] != -1)]
                    legal_from = legal_valid[:, 0]
                    legal_to = legal_valid[:, 1]

                    # Probabilités pour les coups légaux seulement
                    legal_scores = from_prob[b, legal_from] * to_prob[b, legal_to]
                    if torch.isnan(legal_scores).any() or (legal_scores < 0).any():
                        with open("nan_scores.log", "a") as f:
                            f.write("\n--- BAD SCORES ---\n")
                            f.write(f"fen={self.get(('env/env_obs/legal_moves', t))[b]}\n")
                            f.write("------------------\n")
                            legal_scores = torch.ones_like(legal_scores)
                    legal_scores = legal_scores / legal_scores.sum()  # normalisation

                    # Sélection
                    if stochastic:
                        idx = torch.multinomial(legal_scores, 1).item()
                    else:
                        idx = legal_scores.argmax().item()
                    from_list.append(legal_valid[idx, 0])
                    to_list.append(legal_valid[idx, 1])
                else:
                    from_list.append(0)
                    to_list.append(0)

            action_from = torch.tensor(from_list, device=obs.device)
            action_to  = torch.tensor(to_list, device=obs.device)
            self.set(("action", t), torch.stack((action_from, action_to), dim=1))


        if compute_entropy:
            entropy = torch.distributions.Categorical(from_prob).entropy() + \
                    torch.distributions.Categorical(to_prob).entropy()
            self.set((f"{self.prefix}entropy", t), entropy)


    def chess_bot(self,obs):
        from_logits, to_logits = self.model(obs)
        from_probs = torch.softmax(from_logits, dim=-1)
        to_probs = torch.softmax(to_logits, dim=-1)
        return from_probs, to_probs


class PPOPenalty(EpisodicAlgo):
    def __init__(self, cfg):
        super().__init__(cfg, autoreset=True)
        obs_size, _ = self.train_env.get_obs_and_actions_sizes()
        self.train_policy = globals()[cfg.algorithm.policy_type](
            obs_size[0],                                     
            cfg.algorithm.architecture.actor_hidden_size,
        ).with_prefix("current_policy/")
        self.eval_policy = KWAgentWrapper(
            self.train_policy, 
            stochastic=False,
            predict_proba=False,
            compute_entropy=False,
        )
        self.critic_agent = VAgent(
            obs_size[0], cfg.algorithm.architecture.critic_hidden_size
        ).with_prefix("critic/")
        self.old_critic_agent = copy.deepcopy(self.critic_agent).with_prefix("old_critic/")
        self.t_all_critics = TemporalAgent(
            Agents(self.critic_agent, self.old_critic_agent)
        )
        self.old_policy = copy.deepcopy(self.train_policy)
        self.old_policy.with_prefix("old_policy/")
        
        self.t_kl_agent = TemporalAgent(KLAgent(self.old_policy, self.train_policy))

        self.policy_optimizer = setup_optimizer(
            cfg.optimizer, self.train_policy
        )
        self.critic_optimizer = setup_optimizer(
            cfg.optimizer, self.critic_agent
        )