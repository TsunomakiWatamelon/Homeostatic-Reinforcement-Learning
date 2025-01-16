import torch
from environments.base_env import HomeostaticEnvironment

class RiskAversionEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, risky_reward=8, non_risky_reward=2, risky_prob=0.25, energy_threshold=100):
        """
        Environnement pour l'expérience d'aversion au risque.
        :param energy_threshold: Niveau d'énergie à atteindre pour terminer un épisode.
        """
        super().__init__(H, setpoints, weights, exponents, effects)
        self.strating_energy = H.item()
        self.risky_reward = risky_reward
        self.non_risky_reward = non_risky_reward
        self.risky_prob = risky_prob
        self.energy_threshold = energy_threshold  # Seuil d'énergie pour terminer
        self.current_state = 0  # 0: non risqué, 1: risqué

    def step(self, action):
        """
        Effectue une étape dans l'environnement.
        :param action: 0 = rester, 1 = changer d'état
        :return: état, récompense ajustée, terminé, info
        """
        # Mise à jour de l'état courant
        if action == 1:
            self.current_state = 1 - self.current_state  # Basculer entre les états (non risqué et risqué)

        # Ajouter de l'énergie interne en fonction de l'état
        if self.current_state == 0:  # Non risqué
            self.state += self.non_risky_reward
        else:  # Risqué
            energy_gain = self.risky_reward if torch.rand(1).item() < self.risky_prob else 0
            self.state += energy_gain  # Ajout en fonction de la probabilité

        # Calcul de la récompense basée sur le drive
        new_state = self.state
        reward = self.drive.get_reward(new_state)

        # Mettre à jour l'état interne dans l'objet `Drive`
        self.drive.update_state(self.state)

        # Vérifier si l'énergie dépasse le seuil pour terminer
        done = self.state >= self.energy_threshold

        # print(f"state : {self.current_state}, reward : {reward}, energie : {self.drive.internal_state}")
        return self.current_state, reward.item(), done, {}

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.current_state = 0  # Retour à l'état non risqué
        self.state = self.strating_energy
        self.drive.update_state(self.state)
        return self.current_state
