# plot_reward.py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def plot_reward(generation, reward_list, sigmas):
    """
    Plota a recompensa por geração e uma média móvel.

    Args:
        generation (int): Número da geração atual.
        reward_list (list): Lista de melhores recompensas por geração.
        sigmas (list): Lista dos valores de sigma por geração.
    """
    plt.figure(1)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float)

    if len(rewards_tensor) >= 11:
        eval_reward = torch.clone(rewards_tensor[-10:])
        mean_eval_reward = round(torch.mean(eval_reward).item(), 2)
        std_eval_reward = round(torch.std(eval_reward).item(), 2)
        plt.clf()
        plt.title(
            f'Gen #{generation}: Best Reward: {reward_list[-1]:.2f}, Sigma: {sigmas[-1]:.4f}, '
            f'[{mean_eval_reward:.1f}±{std_eval_reward:.1f}]'
        )
    else:
        plt.clf()
        plt.title('Treinando...')

    plt.xlabel('Geração')
    plt.ylabel('Recompensa')
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
