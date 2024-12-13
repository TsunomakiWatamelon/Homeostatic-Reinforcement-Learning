import torch
from models.drive import Drive

class HomeostaticEnvironment:
    def __init__(self, H, setpoints, weights, exponents, effects):
        """
        Initialise l'environnement homéostatique.
        """
        self.state = torch.tensor(H, dtype=torch.float32)  # État interne initial
        self.effects = {key: torch.tensor(value, dtype=torch.float32) for key, value in effects.items()}  # Effets des actions

        # Initialisation de l'objet Drive
        self.drive = Drive(len_state=len(H), optimal_state=setpoints, state_weights=weights, n=exponents[0], m=exponents[1])

        # Mise à jour de l'état interne initial dans Drive
        self.drive.update_state(self.state)

    def update_state(self, action):
        """
        Met à jour l'état interne en fonction de l'action prise.
        """
        if action in self.effects:
            self.state += self.effects[action]
        else:
            raise ValueError(f"Action inconnue : {action}")

    def step(self, action):
        """
        Effectue une étape dans l'environnement en prenant une action.
        """
        # Calcul de l'état après l'action
        current_state = self.state.clone()
        self.update_state(action)
        new_state = self.state.clone()
        
        # Calcul de la récompense
        reward = self.drive.get_reward(new_state)

        # Mise à jour de l'état interne dans l'objet Drive
        self.drive.update_state(new_state)

        # Retourne le nouvel état, la récompense et le signal de fin d'épisode
        return new_state, reward, False, {}
