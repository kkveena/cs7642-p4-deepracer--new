import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from src.agents import Agent
from src.utils import device

class PPOAgent(Agent):
    def __init__(self, environment, gamma=0.99, lr=3e-4, clip_eps=0.2, ent_coef=0.01, epoch_k=10, batch_size=64):
        super().__init__(environment)
        self.device = device()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.epoch_k = epoch_k
        self.batch_size = batch_size

        # Dimensions
        self.obs_dim = environment.observation_space.shape[0]
        self.act_dim = environment.action_space.shape[0]
        
        # ACTION SCALING PARAMETERS
        # We read the physical limits from the environment to scale our output correctly
        self.act_low = torch.tensor(environment.action_space.low, dtype=torch.float32).to(self.device)
        self.act_high = torch.tensor(environment.action_space.high, dtype=torch.float32).to(self.device)
        
        # Actor (Policy) - Outputs Mean in range [-1, 1] via Tanh
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, self.act_dim),
            nn.Tanh() 
        ).to(self.device)

        self.log_std = nn.Parameter(torch.zeros(self.act_dim).to(self.device))

        # Critic (Value)
        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(self.device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], lr=lr)
        self.buffer = []

    def get_action(self, observation, train=True):
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)

        mean = self.actor(observation)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if train:
            raw_action = dist.sample()
        else:
            raw_action = mean

        # 1. Clamp to valid distribution range [-1, 1] for stability
        raw_action = torch.clamp(raw_action, -1.0, 1.0)
        
        log_prob = dist.log_prob(raw_action).sum(axis=-1)

        # 2. UNSCALE ACTION (Map [-1, 1] to [Physical Low, Physical High])
        # Formula: action = low + 0.5 * (raw + 1) * (high - low)
        scaled_action = self.act_low + 0.5 * (raw_action + 1.0) * (self.act_high - self.act_low)

        return scaled_action.cpu().detach().numpy().flatten(), log_prob.cpu().detach().item(), dist.entropy().mean().item()

    def store(self, obs, action, reward, done, log_prob, val):
        self.buffer.append((obs, action, reward, done, log_prob, val))

    def update(self):
        if len(self.buffer) == 0: return
        
        # Unpack buffer
        obs, acts, rews, dones, old_log_probs, vals = zip(*self.buffer)
        
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        # Note: We technically train on the *scaled* actions here if we stored them. 
        # Ideally PPO trains on raw, but for assignment simplicity, this is stable.
        acts = torch.tensor(acts, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Returns
        returns = []
        G = 0
        for r, d in zip(reversed(rews), reversed(dones)):
            if d: G = 0
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Updates
        for _ in range(self.epoch_k):
            # Note: Since we stored Scaled actions, we must Inverse Scale them to get Raw for log_prob
            # or just accept slight drift. For this assignment, we re-run the actor.
            
            # Re-run actor to get current dist
            mean = self.actor(obs)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            
            # We need to calculate log_prob of the actions we took. 
            # Since 'acts' are scaled, we reverse-map them to [-1, 1] to check against distribution
            # Reverse: raw = 2 * (scaled - low) / (high - low) - 1
            raw_acts_approx = 2 * (acts - self.act_low) / (self.act_high - self.act_low) - 1
            raw_acts_approx = torch.clamp(raw_acts_approx, -1.0, 1.0)
            
            new_log_probs = dist.log_prob(raw_acts_approx).sum(axis=-1)
            entropy = dist.entropy().mean()
            v_pred = self.critic(obs).squeeze()
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = returns - v_pred.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(v_pred, returns)
            loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.buffer = []