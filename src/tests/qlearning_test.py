from models.qlearning import QLearning
import numpy as np 

def test_qlearning():
    """
    Teste l'implémentation de Q-learning avec un environnement simple.
    """
    # Configuration de l'environnement fictif
    state_size = 3  # Trois états possibles : 0, 1, 2
    action_size = 2  # Deux actions possibles : 0 et 1
    alpha = 0.1  # Taux d'apprentissage
    gamma = 0.9  # Facteur d'actualisation
    epsilon = 0.1  # Probabilité d'exploration

    # Création explicite de la Q-table
    q_table = {
        0: np.zeros(2),  # État 0 avec deux actions possibles
        1: np.zeros(2),  # État 1 avec deux actions possibles
        2: np.zeros(2),  # État 2 avec deux actions possibles
    }

    # Instancier l'agent 
    agent = QLearning(state_size=3, action_size=2, alpha=0.1, gamma=0.99, epsilon=0.1, q_table=q_table)

    # Table des transitions et récompenses
    transitions = {
        0: {0: (1, 5), 1: (2, 0)},  # Depuis l'état 0, action 0 -> état 1 (reward=5), action 1 -> état 2 (reward=0)
        1: {0: (1, 5), 1: (2, 10)},  # Depuis l'état 1, action 0 -> état 1 (reward=5), action 1 -> état 2 (reward=10)
        2: {0: (2, 0), 1: (0, 1)},  # Depuis l'état 2, action 0 -> état 2 (reward=0), action 1 -> état 0 (reward=1)
    }

    # Paramètres d'entraînement
    episodes = 1000
    max_steps = 100

    for episode in range(episodes):
        state = 0  # Commence toujours à l'état 0
        for step in range(max_steps):
            # Choisir une action
            action = agent.choose_action(state)

            # Obtenir le prochain état et la récompense
            next_state, reward = transitions[state][action]

            # Mettre à jour la table Q
            agent.update_q_value(state, action, reward, next_state)

            # Mise à jour de l'état courant
            state = next_state

    # Vérifier si l'agent a appris à choisir l'action optimale
    optimal_policy = [np.argmax(agent.q_table[s]) for s in range(state_size)]
    expected_policy = [0, 1, 1]  # Attendu : action 0 pour état 0, action 1 pour état 1 et 2

    print("Table Q après apprentissage :")
    print(agent.q_table)
    print("Politique optimale apprise :", optimal_policy)
    print("Politique attendue :", expected_policy)

    # Test de réussite
    assert optimal_policy == expected_policy, "La politique optimale apprise est incorrecte."
    print("Test Q-learning réussi. L'agent apprend correctement à maximiser les récompenses.")

# Exécuter le test
test_qlearning()
