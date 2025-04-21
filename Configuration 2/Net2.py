import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # Couche supplémentaire
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class NetContinousActions(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_dim, hidden_size, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),  # Couche supplémentaire
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )
        self.log_std = nn.Parameter(
            torch.zeros(act_dim)
        )  # learnable log std for Gaussian policy

    def forward(self, obs):
        mean = self.net(obs)
        std = self.log_std.exp()
        return mean, std


# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, state, action, reward, terminated, next_state):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = (state, action, reward, terminated, next_state)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         indices = np.random.choice(len(self.memory), size=batch_size, replace=True)
#         return [self.memory[idx] for idx in indices]
#         # return random.choices(self.memory, k=batch_size)

#     def __len__(self):
#         return len(self.memory)
