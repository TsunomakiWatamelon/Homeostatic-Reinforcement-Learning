import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """
    Réseau pour la politique (Actor) et la valeur (Critic).
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        # Réseau pour l'acteur (politique)
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh()  # Actions normalisées entre -1 et 1
        )
        # Réseau pour le critique (valeur de l'état)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        state_value = self.critic(state)
        return action_mean, state_value


class PPOAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=3e-4, gamma=0.99, clip_epsilon=0.2, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

        self.policy = ActorCritic(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        """
        Sélectionne une action basée sur la politique actuelle.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Ajouter une dimension batch
        action_mean, _ = self.policy(state_tensor)
        dist = Normal(action_mean, torch.ones_like(action_mean) * 0.1)  # Distribution normale
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob.detach()

    def compute_advantages(self, rewards, values, next_values, dones):
        """
        Calcule les avantages via l'équation des TD(λ).
        """
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, log_probs, rewards, next_states, dones):
        """
        Met à jour la politique selon PPO.
        """
        # Conversion
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        log_probs_tensor = torch.FloatTensor(log_probs)
        rewards_tensor = torch.FloatTensor(rewards)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones)

        _, values = self.policy(states_tensor)
        _, next_values = self.policy(next_states_tensor)

        # Advantages
        advantages = self.compute_advantages(rewards_tensor, values.detach().squeeze(), next_values.detach().squeeze(), dones_tensor)

        # Calcul de la politique cible
        for _ in range(10):  # Plusieurs itérations d'optimisation
            action_mean, values = self.policy(states_tensor)
            dist = Normal(action_mean, torch.ones_like(action_mean) * 0.1)
            new_log_probs = dist.log_prob(actions_tensor).sum(dim=1)
            entropy = dist.entropy().mean()

            # Ratio des probabilités
            ratio = torch.exp(new_log_probs - log_probs_tensor)

            # Pertes
            surr1 = ratio * torch.FloatTensor(advantages)
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * torch.FloatTensor(advantages)
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values.squeeze(), rewards_tensor)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            # Optimisation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
