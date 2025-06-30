# PPO code based on pytorch_minimal_ppo
# @misc{pytorch_minimal_ppo,
#     author = {Barhate, Nikhil},
#     title = {Minimal PyTorch Implementation of Proximal Policy Optimization},
#     year = {2021},
#     publisher = {GitHub},
#     journal = {GitHub repository},
#     howpublished = {\url{https://github.com/nikhilbarhate99/PPO-PyTorch}},
# }

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, NN_conf, use_gpu=True):
        super(ActorCritic, self).__init__()

        if NN_conf == 'tanh':
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 64), nn.Tanh(),
                nn.Linear(64, action_dim), nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128), nn.Tanh(),
                nn.Linear(128, 64), nn.Tanh(),
                nn.Linear(64, 1)
            )
        elif NN_conf == 'relu':
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, action_dim), nn.ReLU()
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1)
            )

        self.set_device(use_gpu)
        self.action_var = torch.full((action_dim,), action_std**2).to(self.device)

    def set_device(self, use_gpu=False):
        self.device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory, gready):
        action_mean = self.actor(state)
        if not gready:
            cov_mat = torch.diag(self.action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            return action.detach()
        else:
            return action_mean.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, conf_ppo, use_gpu=False):
        self.lr = conf_ppo['lr']
        self.betas = conf_ppo['betas']
        self.gamma = conf_ppo['gamma']
        self.eps_clip = conf_ppo['eps_clip']
        self.K_epochs = conf_ppo['K_epochs']
        self.entropy_weight = conf_ppo.get('entropy_weight', 0.01)
        self.lam_a = conf_ppo.get('lam_a', 0.0)
        self.normalize_rewards = conf_ppo.get('normalize_rewards', False)

        action_std = conf_ppo['action_std']
        self.set_device(use_gpu)

        self.policy = ActorCritic(state_dim, action_dim, action_std, NN_conf=conf_ppo['nn_type'], use_gpu=use_gpu).to(self.device)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std, NN_conf=conf_ppo['nn_type'], use_gpu=use_gpu).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.loss_a = self.loss_max = self.loss_min = 0.0

    def set_device(self, use_gpu=True, set_policy=False):
        self.device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
        if set_policy:
            self.policy.actor.to(self.device)
            self.policy.critic.to(self.device)
            self.policy.action_var.to(self.device)
            self.policy.set_device(self.device)
            self.policy_old.actor.to(self.device)
            self.policy_old.critic.to(self.device)
            self.policy_old.action_var.to(self.device)
            self.policy_old.set_device(self.device)

    def select_action(self, state, memory, gready=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if not torch.isfinite(state).all():
            print("[🚨] Invalid state detected in select_action:", state)
            state = torch.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)
        return self.policy_old.act(state, memory, gready).cpu().data.numpy().flatten()

    def estimate_action(self, state, action):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = torch.FloatTensor(action.reshape(1, -1)).to(self.device)
        return self.policy_old.evaluate(state, action)
    
    def get_entropy(self):
        return getattr(self, 'last_entropy', 0.0)

    def update(self, memory, to_tensor=False):
#        self.set_device(use_gpu, set_policy=True)

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        if to_tensor:
            memory.states = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.states]
            memory.actions = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.actions]
            memory.logprobs = [torch.FloatTensor(i.reshape(1, -1)).to(self.device) for i in memory.logprobs]

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if self.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            MseLoss = 0.5 * self.MseLoss(state_values, rewards)
            loss = -torch.min(surr1, surr2) + MseLoss - self.entropy_weight * dist_entropy

            if self.lam_a != 0:
                mu = torch.squeeze(torch.stack(memory.actions[:-1]).to(self.device), 1).detach()
                mu_nxt = torch.squeeze(torch.stack(memory.actions[1:]).to(self.device), 1).detach()
                loss += 0.5 * self.MseLoss(mu_nxt, mu) * self.lam_a

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.loss_a = MseLoss.cpu().data.numpy().flatten()[0]
        self.loss_max = advantages.max().cpu().data.numpy().flatten()[0]
        self.loss_min = advantages.min().cpu().data.numpy().flatten()[0]
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.last_entropy = dist_entropy.mean().item()  # 💡 최근 entropy 저장
