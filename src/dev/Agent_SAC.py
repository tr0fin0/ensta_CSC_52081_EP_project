import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
import gymnasium as gym
import gymnasium.wrappers as gym_wrap

# Constants
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6

# Utility functions
def soft_update(target, source, tau):
    """Soft update the target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    """Hard update the target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    """Initialize weights for linear layers."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Q-Network for the critic
class QNetwork(nn.Module):
    def __init__(self, num_channels, height, width, num_actions, hidden_dim, CNN):
        super().__init__()
        # Feature extractor
        self.feature_extractor = CNN((num_channels, height, width), hidden_dim)

        # Q1 architecture
        self.linear1 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(hidden_dim + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        x = self.feature_extractor(state)
        xu = torch.cat([x, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

# Gaussian Policy Network
class GaussianPolicy(nn.Module):
    def __init__(self, num_channels, height, width, num_actions, hidden_dim, CNN):
        super().__init__()
        # Feature extractor
        self.feature_extractor = CNN((num_channels, height, width), hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = self.feature_extractor(state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + EPSILON)  # Enforce action bounds
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

# Deterministic Policy Network
class DeterministicPolicy(nn.Module):
    def __init__(self, num_channels, height, width, num_actions, hidden_dim, CNN):
        super().__init__()
        # Feature extractor
        self.feature_extractor = CNN((num_channels, height, width), hidden_dim)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        self.apply(weights_init_)

    def forward(self, state):
        x = self.feature_extractor(state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

# SAC Agent
class AgentSAC:
    def __init__(
            self,
            state_shape,
            action_shape,
            device,
            model_dir,
            log_dir,
            CNN,
            policy_type="Gaussian",
            gamma=0.99,
            tau=0.005,
            lr=0.0003,
            alpha=0.2,
            auto_alpha_tuning=False,
            batch_size=32,
            hidden_size=256,
            target_update_interval=1,
            load_state=False,
            load_model=None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.lr = lr
        self.device = device
        self.policy_type = policy_type
        self.auto_alpha_tuning = auto_alpha_tuning
        self.target_update_interval = target_update_interval
        self.num_channels, self.height, self.width = state_shape
        self.num_actions = action_shape[0]
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.load_state = load_state
        self.load_model = load_model

        # Initialize networks
        self.critic = QNetwork(self.num_channels, self.height, self.width, self.num_actions, hidden_size, CNN).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(self.num_channels, self.height, self.width, self.num_actions, hidden_size, CNN).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            if self.auto_alpha_tuning:
                self.target_entropy = -torch.prod(torch.Tensor(action_shape).to(self.device)).item()
                self.alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.alpha], lr=self.lr)

            self.policy = GaussianPolicy(self.num_channels, self.height, self.width, self.num_actions, hidden_size, CNN).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
        else:
            self.alpha = 0
            self.auto_alpha_tuning = False
            self.policy = DeterministicPolicy(self.num_channels, self.height, self.width, self.num_actions, hidden_size, CNN).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        # Replay buffer
        self.buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(300000, device=torch.device("cpu"))
        )
        self.updates = 0

        if load_state:
            if load_model is None:
                raise ValueError("Specify a model name for loading.")
            self.load_model = load_model
            self.load(load_model)

    def store(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        state_tensor = torch.tensor(state)
        action_tensor = torch.tensor(action)
        reward_tensor = torch.tensor(reward)
        next_state_tensor = torch.tensor(next_state)
        done_tensor = torch.tensor(done)

        if not (torch.isnan(state_tensor).any() or torch.isnan(action_tensor).any() or torch.isnan(reward_tensor).any() or torch.isnan(next_state_tensor).any()):
            self.buffer.add(
                TensorDict({
                    "state": state_tensor,
                    "action": action_tensor,
                    "reward": reward_tensor,
                    "next_state": next_state_tensor,
                    "done": done_tensor
                }, batch_size=[])
            )

    def get_samples(self, batch_size):
        """Sample a batch of transitions from the replay buffer."""
        batch = self.buffer.sample(batch_size)

        states = batch.get('state').type(torch.FloatTensor).to(self.device)
        next_states = batch.get('next_state').type(torch.FloatTensor).to(self.device)
        actions = batch.get('action').squeeze().to(self.device)
        rewards = batch.get('reward').squeeze().to(self.device)
        dones = batch.get('done').squeeze().to(self.device)

        return states, actions, rewards, next_states, dones

    def get_action(self, state, eval=False):
        """Select an action based on the current policy."""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_network(self, batch_size):
        """Update the networks using a batch of samples."""
        self.updates += 1
        states, actions, rewards, next_states, dones = self.get_samples(batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_states)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rewards + (1 - dones.float()) * self.gamma * min_qf_next_target

        # Compute Q-function losses
        qf1, qf2 = self.critic(states, actions)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        # Update Q-functions
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)  # Retain graph for the next backward pass
        qf2_loss.backward(retain_graph=True)  # Retain graph for the next backward pass
        self.critic_optim.step()

        # Compute policy loss
        pi, log_pi, _ = self.policy.sample(states)
        qf1_pi, qf2_pi = self.critic(states, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update policy
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optim.zero_grad()
        policy_loss.backward(retain_graph=True)  # Retain graph for the next backward pass
        self.policy_optim.step()

        # Update alpha (if automatic temperature tuning is enabled)
        if self.auto_alpha_tuning:
            alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            
            alpha_loss.backward(retain_graph=True)  # Retain graph for alpha loss
            torch.nn.utils.clip_grad_norm_([self.alpha], max_norm=1.0)
            self.alpha_optim.step()
            self.alpha = self.alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        # Update target network
        if self.updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    def save(self, save_name='SAC'):
        """Save the current model to a file."""
        save_path = str(self.model_dir / f"{save_name}_{self.updates}.pt")

        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'alpha': self.alpha,
            'alpha_optim_state_dict': self.alpha_optim.state_dict() if self.auto_alpha_tuning else None,
            'updates': self.updates,
        }, save_path)
        print(f"Model saved to {save_path} at step {self.updates}")

    def load(self, model_name):
        """Load a model from a file."""
        loaded_model = torch.load(str(self.model_dir / model_name), map_location=self.device)

        self.critic.load_state_dict(loaded_model['critic_state_dict'])
        self.critic_target.load_state_dict(loaded_model['critic_target_state_dict'])
        self.policy.load_state_dict(loaded_model['policy_state_dict'])
        self.critic_optim.load_state_dict(loaded_model['critic_optim_state_dict'])
        self.policy_optim.load_state_dict(loaded_model['policy_optim_state_dict'])
        if self.auto_alpha_tuning:
            self.alpha = loaded_model['alpha']
            self.alpha_optim.load_state_dict(loaded_model['alpha_optim_state_dict'])
        self.updates = loaded_model['updates']

    def write_log(self, dates, times, rewards, lengths, losses, log_filename='log_SAC.csv'):
        """Write training logs to a CSV file."""
        rows = [
            ['date'] + dates,
            ['time'] + times,
            ['reward'] + rewards,
            ['length'] + lengths,
            ['loss'] + losses,
        ]
        with open(str(self.log_dir / log_filename), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(rows)