import torch
from environments.base_env import HomeostaticEnvironment

class RiskAversionEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, risky_reward=8, non_risky_reward=2, risky_prob=0.25, env_effect = -2, max_timestep=500):
        """
        Environnement pour l'expérience d'aversion au risque.
        """
        super().__init__(H, setpoints, weights, exponents, effects)
        self.state = 0
        self.initial_state = 0
        self.risky_reward = risky_reward
        self.non_risky_reward = non_risky_reward
        self.risky_prob = risky_prob
        self.max_timestep = max_timestep
        self.current_timestep = 0
        self.env_effects = env_effect
        self.current_markov_state = 0

    def step(self, action, show=False):
        """
        Effectue une étape dans l'environnement.
        :param action: 0 = rester, 1 = changer d'état
        :return: état, récompense ajustée, terminé, q_table_state, timestep
        """
        self.state += self.env_effects

        if action == 1:
            self.current_markov_state = abs(self.current_markov_state - 1) # Permet de changer d'état de 1 à 0 ou de 0 à 1

        # Mise à jour de l'état courant selon l'action
        if self.current_markov_state == 1:
            new_state = self.state + (self.risky_reward if torch.rand(1).item() < self.risky_prob else 0)

        if self.current_markov_state == 0:
            new_state = self.state + self.non_risky_reward

        # Calcul de la récompense basée sur le drive
        new_state = torch.as_tensor(new_state)
        self.drive.update_state(self.state) # Pour mettre à jour l'état interne post effect mais pré action
        reward = self.drive.get_reward(new_state)

        # Mettre à jour l'état interne dans l'objet `Drive` et du module
        self.drive.update_state(new_state)
        self.state = new_state

        # Avancer dans le temps
        self.current_timestep += 1

        # Vérifier si l'épisode est terminé
        done = self.current_timestep >= self.max_timestep

        # print(f"state : {self.current_state}, reward : {reward}, energie : {self.drive.internal_state}")

        # internal state, reward, done, q_table_state, timestep
        return self.state, reward.item(), done, self.current_markov_state, self.current_timestep

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.state = 0
        self.current_markov_state = 0
        self.drive.update_state(self.state)
        self.current_timestep = 0
        
        return self.state, self.current_markov_state, self.current_timestep
