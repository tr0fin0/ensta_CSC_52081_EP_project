import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def plot_reward(episode_num, reward_list, actions, save_dir) -> None:
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
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float).to("cuda")

    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:]).to("cuda")
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
        reward_f = torch.clone(rewards_tensor[:50]).to("cuda")
        means = rewards_tensor.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.ones(49)*torch.mean(reward_f), means))
        plt.plot(means.numpy())
    
    plt.savefig(f"{save_dir}/reward_plot.png")

    #plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)