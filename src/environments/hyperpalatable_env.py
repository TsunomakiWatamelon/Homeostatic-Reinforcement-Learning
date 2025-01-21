from environments.base_env import HomeostaticEnvironment

class HyperpalatableEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, palatability_bonus, normal_food_reward, energy_cost, no_normal_food = False, no_tatsy_food = False):
        """
        Initialise l'environnement pour l'expérience de surconsommation de nourriture hyperpalatable.
        Hérite de HomeostaticEnvironment.
        """
        super().__init__(H, setpoints, weights, exponents, effects)  # Appel du constructeur parent
        self.strating_energy = H[0]
        self.palatability_bonus = palatability_bonus  # Bonus pour nourriture hyperpalatable (T)
        self.normal_food_reward = normal_food_reward  # Récompense pour nourriture normale
        self.energy_cost = energy_cost # L'energie consomé a chaque step

        self.no_normal_food = no_normal_food # can choose between hyperpalatable food and nothing
        self.no_tasty_food = no_tatsy_food # can only choose between normal food and nothing (nul)

    def step(self, action):
        """
        Effectue une étape dans l'environnement en prenant une action.
        """
        # Definir l'action choisit
        if not self.no_normal_food and not self.no_tasty_food:
            normal = action == 0
            hyper = action == 1
        elif self.no_tasty_food:
            normal = action == 0
            hyper = False
        else:
            normal = False
            hyper = action == 0
        
        # Modifier l'état interne 
        self.state[0] -= self.energy_cost # A chaque step il consomme energy cost energie

        if normal:  # Nourriture normale
            self.state[0] += self.normal_food_reward
        elif hyper == 1:  # Nourriture hyperpalatable
            self.state[0] += self.normal_food_reward + self.palatability_bonus

        # Calcule de la récompense 
        new_state = self.state[0]
        reward = self.drive.get_reward(new_state)

        # Mise à jour de l'état interne dans l'objet Drive
        self.drive.update_state(self.state)

        # Retourner l'état, la récompense totale et le signal de fin d'épisode
        base_state = 0 # On a un seul état dans cette experience
        return base_state, reward, False, {}

    def reset(self):
        state = 0  # On a un seul etats
        self.state = [self.strating_energy] # Revenir au niveau d'energie de base
        self.drive.update_state(self.state)
        return state
    
    def set_no_normal_food(self,no_normal_food=True):
        self.no_normal_food = no_normal_food

    def set_no_tasty_food(self,no_tatsy_food=True):
        self.no_tasty_food = no_tatsy_food