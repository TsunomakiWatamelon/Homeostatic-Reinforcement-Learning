from environments.base_env import HomeostaticEnvironment

class AnticipatoryEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, signal_timesteps):
        """
        Environnement pour la réponse anticipatoire.
        :param signal_timesteps: Temps avant l'injection où le signal est donné.
        """
        super().__init__(H, setpoints, weights, exponents, effects)
        self.signal_timesteps = signal_timesteps  # Moments où le signal est donné
        self.current_timestep = 0

    def step(self, action):
        """
        Simule une étape dans l'environnement.
        :param action: 0 = aucune réponse, 1 = réponse anticipatoire
        :return: (nouvel état, récompense, terminé, info)
        """
        # Appliquer l'action (réponse anticipatoire)
        if action == 1:
            self.update_state("anticipate")  # Augmente légèrement la température

        # Vérifier si c'est le moment de l'injection
        if self.current_timestep in self.signal_timesteps:
            self.update_state("injection")

        # Calcul de la récompense
        reward = -self.drive.get_drive()

        # Avancer dans le temps
        self.current_timestep += 1
        done = self.current_timestep > max(self.signal_timesteps)

        return self.state, reward, done, {}

    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.current_timestep = 0
        self.state = self.drive.optimal_state.clone()
        self.drive.update_state(self.state)
        return self.state
