import numpy as np

class QLearning:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Implémente un agent Q-learning.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Taux d'apprentissage
        self.gamma = gamma  # Facteur d'actualisation
        self.epsilon = epsilon  # Probabilité d'exploration

        # Initialisation de la table Q avec des zéros
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        """
        Choisit une action en fonction de la stratégie epsilon-greedy.
        :param state: État courant.
        :return: Action choisie (entier entre 0 et action_size - 1).
        """
        if np.random.rand() < self.epsilon:
            # Exploration : choisir une action aléatoire
            return np.random.randint(self.action_size)
        else:
            # Exploitation : choisir l'action avec la plus grande valeur Q
            return np.argmax(self.q_table[state])

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
        self.q_table = np.zeros((self.state_size, self.action_size))

    def get_q_table(self):
        return self.q_table
