from SkipFrame import SkipFrame

import gymnasium as gym
import gymnasium.wrappers as gym_wrap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def agent_evaluation(agent, environment: gym, seeds: list[int]) -> tuple[list, list]:
    agent.epsilon = 0

    scores = []
    actions = []

    for episode, seed_id in enumerate(seeds):
        state, info = environment.reset(seed=seed_id)
        score = 0
        action = 0

        updating = True
        while updating:
            action = agent.get_action(state)
            state, reward, terminated, truncated, info = environment.step(action)

            updating = not (terminated or truncated)
            score += reward
            action += 1

        scores.append(score)
        actions.append(action)

        print(f"Episode:{episode}, Score:{score:.2f}, actions: {action}")

    environment.close()

    return scores, actions


def get_environment_discrete():
    env_discrete = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=False
    )
    env_discrete = SkipFrame(env_discrete, skip=4)
    env_discrete = gym_wrap.GrayscaleObservation(env_discrete)
    env_discrete = gym_wrap.ResizeObservation(env_discrete, shape=(84, 84))
    env_discrete = gym_wrap.FrameStackObservation(env_discrete, stack_size=4)

    return env_discrete


def plot_reward(episode_num, reward_list, actions) -> None:
    """
    Plots the reward per episode and the moving average of the reward.

    Args:
        episode_num (int): The current episode number.
        reward_list (list): A list of rewards obtained per episode.
        actions (int): The total number of actions taken so far.

    Returns:
        None
    """
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)

    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:])
        mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
        std_eval_reward = round(torch.std(eval_reward).item(), 2)

        plt.clf()
        plt.title(
            f'#{episode_num}: {actions} actions, [{mean_eval_reward:.1f}±{std_eval_reward:.1f}]'
        )
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_tensor.numpy())

    if len(rewards_tensor) >= 50:
        reward_f = torch.clone(rewards_tensor[:50])
        means = rewards_tensor.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(49)*torch.mean(reward_f), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)
