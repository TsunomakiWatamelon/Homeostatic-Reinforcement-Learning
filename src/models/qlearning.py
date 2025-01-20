import numpy as np 
import copy

class QLearning:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, temperature=0.1, q_table=None):
        """
        Implémente un agent Q-learning.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur d'actualisation
        self.temperature = temperature # Température pour la règle softmax

        # Initialisation de la table Q avec des zéros
        if q_table is not None and type(q_table) == dict:
            self.q_table = copy.deepcopy(q_table)
            self.is_dict = True
        else:
            self.q_table = np.zeros((state_size, action_size))
            self.is_dict = False

    def choose_action(self, state, evaluation=False):
        """
        Choisit une action en fonction de la règle softmax.
        :param state: État courant.
        :return: Action choisie (entier entre 0 et action_size - 1).
        """
        # Vérification de l'existence de l'état dans la table Q
        if self.is_dict and state not in self.q_table:
            raise KeyError(f"L'état {state} n'existe pas dans la table Q.")

        # Vérification que des actions sont disponibles
        if not self.is_dict and len(self.q_table[state]) == 0:
            raise ValueError(f"Pas d'actions disponibles pour l'état {state}.")

        q_values = self.q_table[state]

        if not evaluation:
            # Appliquer la règle softmax pour calculer les probabilités
            tau = self.temperature  # Température pour la règle softmax
            exp_q_values = np.exp(q_values / tau)  # Exponentiation des valeurs Q
            probabilities = exp_q_values / np.sum(exp_q_values)  # Distribution des probabilités
        
        else:
            # Softmax rule sans température pour l'évaluation
            exp_q_values = np.exp(q_values)
            probabilities = exp_q_values / np.sum(exp_q_values)

        # Choisir une action en fonction des probabilités calculées
        return np.random.choice(len(q_values), p=probabilities)


    def update_q_value(self, state, action, reward, next_state):
        """
        Met à jour la table Q selon la règle de Q-learning.
        :param state: État courant.
        :param action: Action prise.
        :param reward: Récompense reçue.
        :param next_state: État suivant après l'action.
        """
        # Calcul de la valeur Q cible (TD target)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]

        # Mise à jour de la valeur Q (TD error)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error


    def reset(self):
        """
        Réinitialise la table Q à des zéros.
        """
        if type(self.q_table) == np.ndarray:
            self.q_table.fill(0)
        else:
            for key in self.q_table.keys():
                self.q_table[key].fill(0)

    def get_q_table(self):
        return self.q_table