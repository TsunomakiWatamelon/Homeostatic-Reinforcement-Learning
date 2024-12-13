from environments.base_env import HomeostaticEnvironment
import torch

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

        # Calcul de la récompense (négatif si éloigné du setpoint)
        new_state = self.state.clone()
        reward = self.drive.get_reward(new_state)

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
    
class EthanolInjection():
    def __init__(self, injection_timestep):
        self.injection_timestep = injection_timestep
    
    def ethanol_curve(t, drop=-1.5, recovery_rate=0.2):
        if t <= 0:
            return 0
        return drop * torch.exp(-recovery_rate * t)  # Récupération exponentielle
    
    def get_effect(self, t):
        """
        Renvoie l'effet de l'injection
        """

        return self.ethanol_curve(t - self.injection_timestep)

class ToleranceResponse():
    def __init__(self, response_timestep):
        self.response_timestep = response_timestep
    
    def tolerance_response_curve(t, peak=1.2, peak_time=3, rise_power=0.5, decay_rate=0.2):
        if t <= 0:
            return 0
        elif t <= peak_time:
            # Augmentation exponentielle jusqu'au pic
            return peak * ((t / peak_time) ** rise_power)
        else:
            return peak * torch.exp(-decay_rate * (t - peak_time))  # Décroissance exponentielle

    def get_effect(self, t):
        """
        Returns the effect of the tolerance response
        """
        return self.tolerance_response_curve(t - self.response_timestep)
    

