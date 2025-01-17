import torch
from environments.base_env import HomeostaticEnvironment

class ExtremeDeviationEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, penalty_factor=0.1, deviation_threshold=20):
        """
        Environnement pour l'évitement des déviations extrêmes.
        :param penalty_factor: Facteur de punition pour les déviations extrêmes.
        :param deviation_threshold: Seuil au-delà duquel la punition est appliquée.
        """
        super().__init__(H, setpoints, weights, exponents, effects)
        self.initial_state = self.state.clone().detach()
        self.penalty_factor = penalty_factor
        self.deviation_threshold = deviation_threshold

    def step(self, action):
        """
        Effectue une étape dans l'environnement.
        :param action: Action à prendre
        :return: état, récompense ajustée, terminé, info
        """
        # recuperer le nouveau etats
        new_state = self.state + self.effects[action]

        # Calcul de la récompense basée sur le drive
        reward = self.drive.get_reward(new_state)
        
        # Mise a jour etats interne 
        self.update_state(action=action)
        
        # print("reward :",reward)

        # Appliquer une punition supplémentaire pour les déviations extrêmes
        # deviation = torch.abs(self.state - self.drive.get_optimal_state())
        # if deviation.max() > self.deviation_threshold:
        #     reward -= self.penalty_factor * reward
        if self.state.item() > self.deviation_threshold:
            reward -= self.penalty_factor * reward

        # Mettre à jour l'état interne dans l'objet Drive
        self.drive.update_state(self.state)

        # Vérifier si l'épisode est terminé (optionnel, basé sur une condition de seuil)
        done = False  # Peut être ajusté selon le scénario
        return 0, reward.item(), done, {}

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.state = self.initial_state.clone().detach()  # Retourne à l'état initial
        self.drive.update_state(self.state)
        return 0
