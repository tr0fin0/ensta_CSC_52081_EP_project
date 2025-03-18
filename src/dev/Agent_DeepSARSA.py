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

class DeepSARSA():
    def __init__(
        self,
        environment,
        device,
        directory_models,
        directory_logs,
        CNN,
        gamma = 0.95,
        epsilon = 0.95,
        epsilon_decay = 0.98,
        epsilon_min = 0.02
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.env = environment
        self.shape_state = self.env.observation_space.shape
        self.shape_action = self.env.action_space.n
        self.dir_models = directory_models
        self.dir_logs = directory_logs
        self.device = device

        self.updating_network = CNN(self.shape_state, self.shape_action).float()
        self.updating_network = self.updating_network.to(device=self.device)
        self.frozen_network = CNN(self.shape_state, self.shape_action).float()
        self.frozen_network = self.frozen_network.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.updating_network.parameters(), lr=0.0002)
        self.loss_function = torch.nn.MSELoss()
        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(10000, device=torch.device('cpu'))
        )
        self.updates = 0

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.shape_action)
        else:
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device
                ).unsqueeze(0)
            action_values = self.updating_network(state)
            action = torch.argmax(action_values, axis=1).item()

        return action
    
    def add_sample(self, state, action, reward, next_state, next_action, done):
        """
        Store a sample in the replay buffer.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            next_action (int): The action taken in the next state.
            done (bool): Whether the episode has terminated.
        """
        self.buffer.add(
            TensorDict({
                "state": torch.tensor(state),
                "action": torch.tensor(action),
                "reward": torch.tensor(reward),
                "next_state": torch.tensor(next_state),
                "next_action": torch.tensor(next_action),
                "done": torch.tensor(done)
            }, batch_size=[])
        )
    
    def get_samples(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, new states, and termination flags.
        """
        batch = self.buffer.sample(batch_size)

        states = batch.get('state').type(torch.FloatTensor).to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        next_states = batch.get('next_state').type(torch.FloatTensor).to(self.device)
        next_actions = batch.get('next_action').squeeze().to(self.device)
        dones = batch.get('done').squeeze().to(self.device)
    
        return states, actions, rewards, next_states, next_actions, dones
    
    def update_network(self, batch_size):
        self.updates += 1
        states, actions, rewards, next_states, next_actions, dones = self.get_samples(batch_size)
        action_values = self.updating_network(states)
        current_estimation = action_values[np.arange(batch_size), actions]
        with torch.no_grad():
            next_action_values = self.frozen_network(next_states)
            target_estimation = rewards + (1 - dones.float()) * self.gamma * next_action_values[np.arange(batch_size), next_actions]

        loss = self.loss_function(current_estimation, target_estimation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()

        return current_estimation, loss
    
    def save(self, save_name:str = 'DEEP_SARSA'):
        """
        Save the current model to a file.

        Args:
            save_name (str): The name to use for the saved model file.
        """
        save_path = str(self.dir_models / f"{save_name}_{self.updates}.pt")

        torch.save({
            'upd_model_state_dict': self.updating_network.state_dict(),
            'frz_model_state_dict': self.frozen_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, save_path)
        print(f"Model saved to {save_path} at update {self.updates}")

    def load(self, model_name):
        """
        Load a model from a file.

        Args:
            model_name (str): The name of the model file to load.
        """
        loaded_model = torch.load(str(self.dir_models / model_name), weights_only=False, map_location=torch.device('cpu'))

        updating_network_parameters = loaded_model['upd_model_state_dict']
        frozen_network_parameters = loaded_model['frz_model_state_dict']
        optimizer_parameters = loaded_model['optimizer_state_dict']

        self.updating_network.load_state_dict(updating_network_parameters)
        self.frozen_network.load_state_dict(frozen_network_parameters)
        self.optimizer.load_state_dict(optimizer_parameters)

    def write_log(
            self,
            rewards,
            losses,
            epsilons,
            log_filename='log_DEEP_SARSA.csv'
        ):
        """
        Write training logs to a CSV file.

        Args:
            rewards (list): List of rewards for each episode.
            losses (list): List of losses for each episode.
            epsilons (list): List of epsilon values for each episode.
            log_filename (str, optional): The name of the log file. Defaults to 'log_DEEP_SARSA.csv'.
        """
        rows = [
            ['reward'] + rewards,
            ['loss'] + losses,
            ['epsilon'] + epsilons
        ]
        with open(str(self.dir_logs / log_filename), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)
        