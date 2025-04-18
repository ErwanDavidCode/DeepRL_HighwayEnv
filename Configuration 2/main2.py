# main_racetrack.py
import gymnasium as gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from configuration2 import config_dict

from Net2 import Net, NetContinousActions
from torch.distributions import Normal

from gymnasium import Wrapper




class PPOContinuous:
    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, gamma=0.99, lambda_=0.95):
        self.actor = NetContinousActions(obs_dim, 128, act_dim)
        self.critic = Net(obs_dim, 128, 1)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps_clip = 0.2

        self.reset_episode()

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze().numpy(), log_prob.item()

    def reset_episode(self):
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'next_states': []
        }

    def store_transition(self, state, action, log_prob, reward, done, next_state):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['next_states'].append(next_state)

    def compute_gae(self):
        rewards = self.memory['rewards']
        dones = self.memory['dones']
        states = torch.tensor(self.memory['states'], dtype=torch.float32)
        next_states = torch.tensor(self.memory['next_states'], dtype=torch.float32)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()

        gae = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        return advantages, returns

    def update(self):
        states = torch.tensor(self.memory['states'], dtype=torch.float32)
        actions = torch.tensor(self.memory['actions'], dtype=torch.float32)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32)

        advantages, returns = self.compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(5):
            mean, std = self.actor(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()





class CustomRewardWrapper(Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        base_env = self.env.unwrapped
        vehicle = base_env.vehicle

        reward = 0.0

        # Pénalité collision
        if vehicle.crashed:
            reward += self.config.get("collision_reward", -1)

        # Coût de recentrage sur la route
        if hasattr(vehicle, 'on_road'):
            reward -= self.config.get("lane_centering_cost", 4) * (1 - vehicle.on_road)

        # Pénalité d'action (encourage les petites actions)
        reward += self.config.get("action_reward", -0.3) * np.square(action).sum()

        return obs, reward, terminated, truncated, info



# Création de l'environnement
env = gym.make('racetrack-v0', render_mode="rgb_array")
env.unwrapped.configure(config_dict)
# On Wrappe l'environnement pour redéfinir la récompense	
env = CustomRewardWrapper(env, config_dict)
env.reset()

# Afficher les états et actions possibles
print("Observation Space:", env.observation_space)
#print("Sample Observation:", env.observation_space.sample())
print("Action Space:", env.action_space)



# === Train ===
obs_dim = int(np.prod(env.observation_space.shape))
act_dim = env.action_space.shape[0]
agent = PPOContinuous(obs_dim, act_dim, actor_lr=3e-4, critic_lr=1e-3)

rewards_all = []
actor_losses = []
critic_losses = []

n_episodes = 10

for episode in range(n_episodes):
    state, _ = env.reset()
    agent.reset_episode()
    total_reward = 0
    done = False

    while not done:
        state_flat = np.array(state).flatten()
        action, log_prob = agent.get_action(state_flat)
        action = np.array([action])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state_flat = np.array(next_state).flatten()

        agent.store_transition(state_flat, action, log_prob, reward, done, next_state_flat)
        state = next_state
        total_reward += reward

    actor_loss, critic_loss = agent.update()
    rewards_all.append(total_reward)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)

    print(f"Ep {episode}: reward = {total_reward:.2f}, actor loss = {actor_loss:.4f}, critic loss = {critic_loss:.4f}")

# === Save model ===
torch.save(agent.actor.state_dict(), "ppo_actor_continuous.pth")
torch.save(agent.critic.state_dict(), "ppo_critic_continuous.pth")

# === Plots ===
plt.plot(rewards_all)
plt.title("Rewards per Episode")
plt.savefig("ppo_racetrack_rewards.png")
plt.clf()

plt.plot(actor_losses)
plt.title("Actor Loss")
plt.savefig("ppo_racetrack_actor_loss.png")
plt.clf()

plt.plot(critic_losses)
plt.title("Critic Loss")
plt.savefig("ppo_racetrack_critic_loss.png")
plt.clf()
