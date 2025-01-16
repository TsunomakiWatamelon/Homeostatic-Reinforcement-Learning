from environments.base_env import HomeostaticEnvironment

class HyperpalatableEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, palatability_bonus, normal_food_reward):
        """
        Initialise l'environnement.
        """
        super().__init__(H, setpoints, weights, exponents, effects)  
        self.palatability_bonus = palatability_bonus  # Bonus de récompense pour nourriture hyperpalatable
        self.normal_food_reward = normal_food_reward  # Récompense pour nourriture normale

    def step(self, action):
        """
        Effectue une étape dans l'environnement.
        """
        # Récompenses spécifiques aux actions
        if action == 0:  # Nourriture normale
            action_reward = self.normal_food_reward
        elif action == 1:  # Nourriture hyperpalatable
            action_reward = self.normal_food_reward + self.palatability_bonus
        else:
            raise ValueError(f"Action inconnue : {action}")

        # Mise à jour de l'état interne via la méthode parent
        new_state, homoeostatic_reward, done, info = super().step(action)

        # Ajout des récompenses spécifiques aux actions
        total_reward = action_reward + homoeostatic_reward

        # Retourne l'état, la récompense totale et les autres informations
        return new_state, total_reward, done, info
