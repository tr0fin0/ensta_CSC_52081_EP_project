import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import math
import numpy as np
import pandas as pd
from pathlib import Path
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional
import csv
import datetime

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.ion()


import seaborn as sns
from tqdm.notebook import tqdm

from IPython.display import Video
from ipywidgets import interact

import warnings

from IPython.display import Video
from pathlib import Path
from typing import List



DIRECTORY_OUTPUT = "output"
DIRECTORY_MODELS = Path(f"{DIRECTORY_OUTPUT}/models/")
DIRECTORY_FIGURES = Path(f"{DIRECTORY_OUTPUT}/images/")
DIRECTORY_LOGS = Path(f"{DIRECTORY_OUTPUT}/logs/")

if not DIRECTORY_FIGURES.exists():
    DIRECTORY_FIGURES.mkdir(parents=True)

if not DIRECTORY_MODELS.exists():
    DIRECTORY_MODELS.mkdir(parents=True)

if not DIRECTORY_LOGS.exists():
    DIRECTORY_LOGS.mkdir(parents=True)



warnings.filterwarnings("ignore", category=UserWarning)


def save_model(policy, iteration):
    """Save the model parameters."""
    model_path = DIRECTORY_MODELS / f"model_iter_{iteration}.pt"
    torch.save(policy.state_dict(), model_path)

def save_log(iteration, rewards, sigmas):
    """Save the logs in a CSV file."""
    log_path = DIRECTORY_LOGS / "training_log.csv"
    data = {"iteration": iteration, "reward": rewards[-1], "sigma": sigmas[-1]}

    df = pd.DataFrame([data])
    if log_path.exists():
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, mode="w", header=True, index=False)


class ObjectiveFunction:
    """
    Objective function for evaluating a policy in the given environment.

    This function runs the policy for a number of episodes and returns the negative of
    the average reward (to transform the problem into a minimization task).

    Parameters
    ----------
    env : gym.Env
        The environment (CarRacing-v3) in which to evaluate the policy.
    policy : nn.Module
        The policy to evaluate.
    num_episodes : int, optional
        Number of episodes per evaluation.
    max_time_steps : float, optional
        Maximum time steps per episode.
    minimization_solver : bool, optional
        If True, returns negative reward for minimization.
    """
    def __init__(self, env: gym.Env, policy: nn.Module, num_episodes: int = 1,
                 max_time_steps: float = float("inf"), minimization_solver: bool = True):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimization_solver = minimization_solver
        self.num_evals = 0

    def eval(self, policy_params: np.ndarray, num_episodes: Optional[int] = None,
             max_time_steps: Optional[float] = None) -> float:
        """
        Evaluate the policy with the given parameters.

        Parameters
        ----------
        policy_params : np.ndarray
            The parameters to evaluate.
        num_episodes : int, optional
            Number of episodes for evaluation.
        max_time_steps : float, optional
            Maximum time steps per episode.

        Returns
        -------
        float
            The (possibly negated) average total reward.
        """
        self.policy.set_params(policy_params)
        self.num_evals += 1

        if num_episodes is None:
            num_episodes = self.num_episodes
        if max_time_steps is None:
            max_time_steps = self.max_time_steps

        total_reward_sum = 0.0
        for _ in range(num_episodes):
            observation, info = self.env.reset()
            episode_reward = 0.0
            for t in range(int(max_time_steps)):
                action = self.policy(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            total_reward_sum += episode_reward

        avg_reward = total_reward_sum / num_episodes

        # Convert to a minimization problem if required.
        if self.minimization_solver:
            avg_reward *= -1.0
        return avg_reward

    def __call__(self, policy_params: np.ndarray, num_episodes: Optional[int] = None,
                 max_time_steps: Optional[float] = None) -> float:
        return self.eval(policy_params, num_episodes, max_time_steps)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_reward(iteration, reward_list, sigmas):
    """
    Plot the reward per generation and a moving average.

    Args:
        generation (int): Current generation number.
        reward_list (list): List of best rewards per generation.
        sigmas (list): List of sigma values per generation.
    """
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)

    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:])
        mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
        std_eval_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(
            f'Iter #{iteration}: Best Reward: {reward_list[-1]:.2f}, Sigma: {sigmas[-1]:.4f}, '
            f'[{mean_eval_reward:.1f}±{std_eval_reward:.1f}]'
        )
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())

    if len(rewards_tensor) >= 50:
        reward_f = torch.clone(rewards_tensor[:50])
        means = rewards_tensor.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(49) * torch.mean(reward_f), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)




def cem_uncorrelated(
    objective_function: Callable[[np.ndarray], float],
    mean_array: np.ndarray,
    var_array: np.ndarray,
    max_iterations: int = 500,
    sample_size: int = 50,
    elite_frac: float = 0.2,
    print_every: int = 10,
    success_score: float = float("inf"),
    num_evals_for_stop: Optional[int] = None,
    hist_dict: Optional[dict] = None,
    policy: Optional[nn.Module] = None,
    save_interval: int = 50
) -> np.ndarray:
    """
    Cross-Entropy Method (CEM) optimization.

    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The objective function to evaluate policies.
    mean_array : np.ndarray
        Initial mean parameters.
    var_array : np.ndarray
        Initial variance for each parameter.
    max_iterations : int, optional
        Maximum number of iterations.
    sample_size : int, optional
        Number of candidate samples per iteration.
    elite_frac : float, optional
        Fraction of top-performing samples to use for updating.
    print_every : int, optional
        Frequency of printing and plotting progress.
    success_score : float, optional
        Score threshold for early stopping.
    num_evals_for_stop : Optional[int], optional
        If provided, evaluate the mean parameters every iteration.
    hist_dict : Optional[dict], optional
        Dictionary to store history.

    Returns
    -------
    np.ndarray
        The optimized mean parameters.
    """
    assert 0.0 < elite_frac <= 1.0
    n_elite = math.ceil(sample_size * elite_frac)

    # Lists for real-time plotting
    reward_list = []
    sigma_list = []

    for iteration_index in range(max_iterations):
        # Sample new candidate solutions from the multivariate normal distribution.
        x_array = np.random.randn(sample_size, mean_array.shape[0]) * np.sqrt(var_array) + mean_array

        # Evaluate each candidate solution.
        score_array = np.array([objective_function(x) for x in x_array])
        sorted_indices = np.argsort(score_array)
        elite_indices = sorted_indices[:n_elite]
        elite_x_array = x_array[elite_indices]

        # Update mean and variance based on the elite samples.
        mean_array = np.mean(elite_x_array, axis=0)
        var_array = np.var(elite_x_array, axis=0)
        score = np.mean(score_array[elite_indices])

        # Append values for plotting.
        reward_list.append(score)
        sigma_list.append(np.mean(var_array))

        if iteration_index % print_every == 0:
            print(f"Iteration {iteration_index}\tScore {score}")
            plot_reward(iteration_index, reward_list, sigma_list)

        if iteration_index % save_interval == 0:
            save_model(policy, iteration_index)

        save_log(iteration_index, reward_list, sigma_list)

        if hist_dict is not None:
            hist_dict[iteration_index] = [score] + mean_array.tolist() + var_array.tolist()

        if num_evals_for_stop is not None:
            score = objective_function(mean_array)
        if score <= success_score:
            break

    return mean_array