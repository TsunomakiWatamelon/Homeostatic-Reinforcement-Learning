from environments.base_env import HomeostaticEnvironment

class HyperpalatableEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, palatability_bonus, normal_food_reward, threshold):
        """
        Initialise l'environnement pour l'expérience de surconsommation de nourriture hyperpalatable.
        Hérite de HomeostaticEnvironment.
        """
        super().__init__(H, setpoints, weights, exponents, effects)  # Appel du constructeur parent
        self.strating_energy = H[0]
        self.palatability_bonus = palatability_bonus  # Bonus pour nourriture hyperpalatable (T)
        self.normal_food_reward = normal_food_reward  # Récompense pour nourriture normale
        self.threshold = threshold  # Seuil pour terminer l'épisode

    def step(self, action):
        """
        Effectue une étape dans l'environnement en prenant une action.
        Calcule la récompense selon la formule donnée et met à jour l'état.
        """
        # Modifier l'état interne en fonction de l'action
        if action == 0:  # Nourriture normale
            self.state[0] += self.normal_food_reward
        elif action == 1:  # Nourriture hyperpalatable
            self.state[0] += self.normal_food_reward + self.palatability_bonus
        else:
            raise ValueError(f"Action inconnue : {action}")

        # Calcule de la récompense 
        new_state = self.state[0]
        reward = self.drive.get_reward(new_state)

        # Mise à jour de l'état interne dans l'objet Drive
        self.drive.update_state(self.state)

        # Vérifier le seuil pour terminer l'épisode est atteint
        done = self.state[0] > self.threshold

        # Retourner l'état, la récompense totale et le signal de fin d'épisode
        base_state = 0 # On a un seul état dans cette experience
        return base_state, reward, done, {}

    def reset(self):
        state = 0  # On a un seul etats
        self.state = [self.strating_energy] # Revenir au niveau d'energie de base
        self.drive.update_state(self.state)
        return state