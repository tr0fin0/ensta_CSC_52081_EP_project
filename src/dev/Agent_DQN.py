import csv
import numpy as np
import os
import torch

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class Agent_DQN:
    """
    A Deep Q-Network (DQN) agent for reinforcement learning.

    Attributes:
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy policy.
        epsilon_decay (float): Decay rate for exploration probability.
        epsilon_min (float): Minimum exploration rate.
        shape_state (tuple): Shape of the state space.
        shape_action (int): Number of possible actions.
        load_state (bool): Whether to load a pre-trained model.
        use_DDQN (bool): Whether to use Double DQN.
        dir_models (str): Directory to save/load models.
        dir_logs (str): Directory to save logs.
        device (torch.device): Device to run the model on (CPU or GPU).
        updating_network (torch.nn.Module): Neural network for updating Q-values.
        frozen_network (torch.nn.Module): Neural network for target Q-values.
        optimizer (torch.optim.Optimizer): Optimizer for training the updating network.
        loss_function (torch.nn.Module): Loss function for training.
        buffer (TensorDictReplayBuffer): Replay buffer for storing experiences.
        actions_taken (int): Counter for the number of actions taken.
        updates (int): Counter for the number of network updates.
        load_model (str): Name of the model to load (if load_state is True).

    Methods:
        store(state, action, reward, new_state, terminated):
            Store a transition in the replay buffer.
        get_samples(batch_size):
            Sample a batch of transitions from the replay buffer.
        get_action(state):
            Select an action based on the current policy.
        update_network(batch_size):
            Update the Q-network using a batch of transitions.
        save(dir_models, save_name):
            Save the current model to a file.
        load(load_dir, model_name):
            Load a model from a file.
        write_log(dates, times, rewards, lengths, losses, epsilons, log_filename='default_log.csv'):
            Write training logs to a CSV file.
    """
    def __init__(
            self,
            shape_state,
            shape_action,
            device,
            directory_models,
            directory_logs,
            CNN,
            load_state = False,
            load_model = None,
            use_DDQN = False,
            gamma = 0.95,
            epsilon = 1,
            epsilon_decay = 0.9999925,
            epsilon_min = 0.05,
        ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.shape_state = shape_state
        self.shape_action = shape_action
        self.load_state = load_state
        self.use_DDQN = use_DDQN
        self.dir_models = directory_models
        self.dir_logs = directory_logs
        self.device = device

        self.updating_network = CNN(self.shape_state, self.shape_action).float()
        self.updating_network = self.updating_network.to(device=self.device)
        self.frozen_network = CNN(self.shape_state, self.shape_action).float()
        self.frozen_network = self.frozen_network.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.updating_network.parameters(), lr=0.0002)
        self.loss_function = torch.nn.SmoothL1Loss()
        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(300000, device=torch.device("cpu"))
        )
        self.actions_taken = 0
        self.updates = 0
 
        if load_state:
            if load_model == None:
                raise ValueError(f"Specify a model name for loading.")

            self.load_model = load_model
            self.load(load_model)


    def store(self, state, action, reward, new_state, terminated):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            new_state (np.ndarray): The next state.
            terminated (bool): Whether the episode has terminated.
        """
        self.buffer.add(
            TensorDict({
                "state": torch.tensor(state),
                "action": torch.tensor(action),
                "reward": torch.tensor(reward),
                "new_state": torch.tensor(new_state),
                "terminated": torch.tensor(terminated)
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
        new_states = batch.get('new_state').type(torch.FloatTensor).to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        terminations = batch.get('terminated').squeeze().to(self.device)
    
        return states, actions, rewards, new_states, terminations

    def get_action(self, state):
        """
        Select an action based on the current policy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            int: The index of the action to take.
        """
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        self.actions_taken += 1

        return action

    def update_network(self, batch_size):
        """
        Update the Q-network using a batch of transitions.

        Args:
            batch_size (int): The number of transitions to sample for the update.

        Returns:
            tuple: A tuple containing the estimated Q-values and the loss value.
        """
        self.updates += 1
        states, actions, rewards, new_states, terminations = self.get_samples(batch_size)
        action_values = self.updating_network(states)
        current_estimation = action_values[np.arange(batch_size), actions]

        if self.use_DDQN:
            with torch.no_grad():
                next_actions = torch.argmax(self.updating_network(new_states), axis=1)
                tar_action_values = self.frozen_network(new_states)
            target_estimation = rewards + (1 - terminations.float()) * self.gamma \
                * tar_action_values[np.arange(batch_size), next_actions]
        else:
            with torch.no_grad():
                tar_action_values = self.frozen_network(new_states)
            target_estimation = rewards + (1 - terminations.float()) * self.gamma \
                * tar_action_values.max(1)[0]

        loss = self.loss_function(current_estimation, target_estimation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.item()

        return current_estimation, loss

    def save(self, save_name:str = 'DQN'):
        """
        Save the current model to a file.

        Args:
            save_name (str): The name to use for the saved model file.
        """
        save_path = str(self.dir_models / f"{save_name}_{self.actions_taken}.pt")

        torch.save({
            'upd_model_state_dict': self.updating_network.state_dict(),
            'frz_model_state_dict': self.frozen_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_number': self.actions_taken,
            'epsilon': self.epsilon
        }, save_path)
        print(f"Model saved to {save_path} at step {self.actions_taken}")

    def load(self, model_name):
        """
        Load a model from a file.

        Args:
            model_name (str): The name of the model file to load.
        """
        loaded_model = torch.load(str(self.dir_models / model_name), weights_only=False)

        updating_network_parameters = loaded_model['upd_model_state_dict']
        frozen_network_parameters = loaded_model['frz_model_state_dict']
        optimizer_parameters = loaded_model['optimizer_state_dict']

        self.updating_network.load_state_dict(updating_network_parameters)
        self.frozen_network.load_state_dict(frozen_network_parameters)
        self.optimizer.load_state_dict(optimizer_parameters)

        if self.load_state == 'eval':
            self.updating_network.eval()
            self.frozen_network.eval()
            self.epsilon_min = 0
            self.epsilon = 0
        elif self.load_state == 'train':
            self.updating_network.train()
            self.frozen_network.train()
            self.actions_taken = loaded_model['action_number']
            self.epsilon = loaded_model['epsilon']
        else:
            raise ValueError(f"Unknown load state. Should be either 'eval' or 'train'.")

    def write_log(
            self,
            dates,
            times,
            rewards,
            lengths,
            losses,
            epsilons,
            log_filename='log_DQN.csv'
        ):
        """
        Write training logs to a CSV file.

        Args:
            dates (list): List of dates for each episode.
            times (list): List of times for each episode.
            rewards (list): List of rewards for each episode.
            lengths (list): List of lengths for each episode.
            losses (list): List of losses for each episode.
            epsilons (list): List of epsilon values for each episode.
            log_filename (str, optional): The name of the log file. Defaults to 'log_DQN.csv'.
        """
        rows = [
            ['date'] + dates,
            ['time'] + times,
            ['reward'] + rewards,
            ['length'] + lengths,
            ['loss'] + losses,
            ['epsilon'] + epsilons
        ]
        with open(str(self.dir_logs / log_filename), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)