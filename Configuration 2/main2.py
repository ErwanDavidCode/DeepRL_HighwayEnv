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
    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, gamma=0.99, lambda_=0.95, eps_clip=0.2):
        self.actor = NetContinousActions(obs_dim, 128, act_dim)
        self.critic = Net(obs_dim, 128, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps_clip = eps_clip

        self.reset_episode()

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.actor(state_tensor)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze().cpu().numpy(), log_prob.item()

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
        # Convertir les états et actions en tableaux NumPy avant de les stocker
        self.memory['states'].append(np.array(state, dtype=np.float32))
        self.memory['actions'].append(np.array(action, dtype=np.float32))
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
        self.memory['next_states'].append(np.array(next_state, dtype=np.float32))

    def compute_gae(self):
        # Convertir les listes en tableaux NumPy pour accélérer la création des tenseurs
        rewards = np.array(self.memory['rewards'], dtype=np.float32)
        dones = np.array(self.memory['dones'], dtype=np.float32)
        states = torch.tensor(np.stack(self.memory['states']), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack(self.memory['next_states']), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()

        gae = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * next_values[t] - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values
        return advantages, returns

    def update(self, advantages, returns, batch_size=64):
        # Préparer les données
        states = torch.tensor(np.stack(self.memory['states']), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack(self.memory['actions']), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(self.memory['log_probs'], dtype=torch.float32, device=self.device)#

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages = advantages.to(self.device)#
        returns = returns.to(self.device)#

        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, advantages, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Entraîner l'actor et le critic par mini-batches
        for _ in range(5):
            for batch in dataloader:
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch

                # Entraîner l'actor
                mean, std = self.actor(batch_states)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5)
                self.actor_optimizer.step()

                # Entraîner le critic
                values = self.critic(batch_states).view(-1)
                batch_returns = batch_returns.view(-1)
                critic_loss = nn.MSELoss()(values, batch_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5)
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
            reward += self.config.get("collision_reward")

        # Valeur qui récompense être sur la route et pénalise être hors route
        reward += self.config.get("on_road_reward") * vehicle.on_road
        
        # # Valeur qui pénalise être hors route
        # reward += self.config.get("off_road_reward") * (1 - vehicle.on_road)

        # Pénalité d'action (encourage les petites actions)
        reward += self.config.get("action_reward") * np.square(action).sum()

        # # # ➕ Pénalité de distance au centre de la voie
        # if vehicle.lane is not None:
        #     lane_center_offset = abs(vehicle.lane.distance(vehicle.position))
        #     reward += self.config.get("lane_centering_reward", 0.2) * (1 - lane_center_offset)

        # Récompense pour éviter les collisions (bonus à la fin de l'épisode)
        if terminated or truncated:
            if not vehicle.crashed:
                reward += self.config.get("no_collision_reward", 10)

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
hyperparams = {
    "actor_lr": 1e-4,        # Reduced from 3e-4 for more stability
    "critic_lr": 5e-4,       # Reduced from 1e-3
    "gamma": 0.99,           # Keep this
    "lambda_": 0.95,         # Keep this
    "eps_clip": 0.1          # Reduced from 0.2 for more conservative updates
}

agent = PPOContinuous(obs_dim, act_dim, **hyperparams)

rewards_all = []
actor_losses = []
critic_losses = []

n_episodes = 200

for episode in range(n_episodes):
    state, _ = env.reset()
    agent.reset_episode()
    total_reward = 0
    done = False

    while not done:
        state_flat = np.array(state).flatten()
        action, log_prob = agent.get_action(state_flat)
        action = np.array([action])  # pour env.step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state_flat = np.array(next_state).flatten()

        # Stocker les transitions
        agent.store_transition(state_flat, action, log_prob, reward, done, next_state_flat)
        state = next_state
        total_reward += reward

    # Calculer les avantages et les retours une seule fois
    advantages, returns = agent.compute_gae()

    # Passer les données en mini-batches à update
    actor_loss, critic_loss = agent.update(advantages, returns)

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
