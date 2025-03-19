from collections import deque
from ipywidgets import interact
from IPython.display import Video
from pathlib import Path
from tqdm.notebook import tqdm
from typing import cast, List, Tuple, Deque, Optional, Callable
import os
import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import random
from CNN import CNN
from SkipFrame import SkipFrame
import numpy as np
# from ReplayBuffer import ReplayBuffer
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import csv

from PPO import PolicyNetwork, ValueNetwork
import torch.distributions as dist


class PPO():
    def __init__(
        self,
        env: gym.Env,
        device: torch.device,
        dir_models:str,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        clip_ratio: float = 0.1,
        max_grad_norm: float = 0.5,
        lam: float = 0.97,
        epochs: int=10,
        batch_size: int = 128,
        buffer_size: int = 1000,
    ):
        self.env = env
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.old_log_probs = []
        self.dir_models = dir_models

        self.policy = PolicyNetwork(
            input_dimensions=self.env.observation_space.shape,
            output_dim=self.env.action_space.shape[0]
        ).to(self.device)

        self.value = ValueNetwork(
            input_dimensions=self.env.observation_space.shape,
            output_dim=1
        ).to(self.device)

        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=self.learning_rate)
        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(self.buffer_size, device=torch.device('cpu'))
        )
        self.updates = 0

    def take_action(self, state):
        mu, std = self.policy(state)
        distribution = dist.Normal(mu, std)
        action = distribution.sample()
        action[:, 0] = torch.clamp(action[:, 0], -1.0, 1.0)  # Steering: [-1, 1]
        action[:, 1:] = torch.clamp(action[:, 1:], 0.0, 1.0) # Gas and brake: [0, 1]
        log_prob = distribution.log_prob(action).sum(dim=-1)  # Compute log probability

        return action, log_prob

    
    def add_sample(self, state, action, reward, next_state, done, log_prob):
        self.buffer.add(
            TensorDict({
                "state": torch.tensor(state),
                "action": torch.tensor(action),
                "reward": torch.tensor(reward),
                "next_state": torch.tensor(next_state),
                "done": torch.tensor(done),
                "log_prob": log_prob.detach(),  # Store log_prob in buffer
            }, batch_size=[])
        )
    
    def get_samples(self, batch_size:int) -> tuple:
        batch = self.buffer.sample(batch_size)
        states = batch.get('state').type(torch.FloatTensor).to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        next_states = batch.get('next_state').type(torch.FloatTensor).to(self.device)
        dones = batch.get('done').squeeze().to(self.device)
        log_probs = batch.get('log_prob').squeeze().to(self.device)

        return states, actions, rewards, next_states, dones, log_probs


    def update(self):
        self.updates += 1
        states, actions, rewards, next_states, dones, old_log_probs = self.get_samples(self.batch_size)
        
        next_states = next_states.squeeze(1)
        states = states.squeeze(1)

        value_loss = self.update_value(states, rewards, next_states, dones)
        policy_loss = self.update_policy(states, actions, old_log_probs, rewards, next_states, dones)

        return value_loss, policy_loss

    def update_value(self, states, rewards, next_states, dones):
        with torch.no_grad():
            next_values = self.value(next_states)
            targets = rewards + self.gamma * next_values * (1 - dones.float())

        values = self.value(states)
        value_loss = (values - targets).pow(2).mean()

        self.optimizer_value.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.optimizer_value.step()

        return value_loss.item()

    def update_policy(self, states, actions, old_log_probs, rewards, next_states, dones):
        mu, std = self.policy(states)
        distribution = torch.distributions.Normal(mu, std)

        log_probs = distribution.log_prob(actions).sum(dim=-1)  # Compute new log probs
        ratio = (log_probs - old_log_probs).exp()  # PPO ratio

        with torch.no_grad():
            next_values = self.value(next_states)
            values = self.value(states)
            advantages = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t].float()) - values[t]
                advantages[t] = gae = delta + self.gamma * self.lam * (1 - dones[t].float()) * gae

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        entropy = distribution.entropy().mean()
        policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy  # Entropy bonus

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer_policy.step()

        return policy_loss.item()
    
    def save(self, save_name: str = 'PPO'):
        path = self.dir_models / f"{save_name}_{self.updates}.pt"
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_value": self.optimizer_value.state_dict(),
            "learning_rate": self.learning_rate,
        }, path)
        print(f"Model saved to {path} at update {self.updates}")
    
    def load(self, load_name: str):
        path = self.dir_models / f"{load_name}.pt"
        model = torch.load(path, map_location=torch.device('cpu'))
        self.policy.load_state_dict(model["policy"])
        self.value.load_state_dict(model["value"])
        self.optimizer_policy.load_state_dict(model["optimizer_policy"])
        self.optimizer_value.load_state_dict(model["optimizer_value"])
        self.learning_rate = model["learning_rate"]
        print(f"Model loaded from {path}")