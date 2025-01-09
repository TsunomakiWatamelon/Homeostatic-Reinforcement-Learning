from environments.base_env import HomeostaticEnvironment
import torch

class AnticipatoryEnvironment(HomeostaticEnvironment):
    def __init__(self, H, setpoints, weights, exponents, effects, signal_timesteps, injection_timesteps, max_timestep):
        """
        Environnement pour la réponse anticipatoire.
        :param signal_timesteps: Temps avant l'injection où le signal est donné.
        """
        super().__init__(H, setpoints, weights, exponents, effects)
        self.initial_state = self.state.clone()
        self.signal_timesteps = signal_timesteps  # Moments où le signal est donné
        self.injection_timesteps = injection_timesteps # Moments où l'injection est donnée
        self.current_timestep = 0
        self.anticipations = []
        self.injections = []
        self.markov_states = [0, 1, 2] # 1 = signal, 2 = post-injection anticipe, 3 = post-injection non anticipe
        self.current_markov_state = 0
        self.max_timestep = max_timestep

    def step(self, action, extinction_trial=False):
        """
        Simule une étape dans l'environnement.

        :param action: Action prise par l'agent.

        :return: Tuple contenant :
        - le nouvel état
        - la récompense
        - un booléen indiquant si l'épisode est terminé
        - l'état markovien courant
        - le timestep courant
        """
        # Appliquer l'action (réponse anticipatoire)
        action_taken = None
        event = None


        if action == 1 and self.current_markov_state == 0:
            action_taken = "anticipate"
            self.current_markov_state = 1 # Anticipation
        if self.current_markov_state == 0 and action == 0:
            self.current_markov_state = 2 # Pas d'anticipation
        if self.current_timestep in self.signal_timesteps:
            event = "injection" # Injection d'éthanol

        # Mettre à jour les conditions de l'environnement
        self.update_conditions(action_taken, event, extinction_trial)
        self.update_state()

        # Calcul de la récompense (négatif si éloigné du setpoint)
        new_state = self.state.clone()
        reward = self.drive.get_reward(new_state)

        # Avancer dans le temps
        self.current_timestep += 0.5 # 0.5 == 30 minutes == 0.5
        done = self.current_timestep >= self.max_timestep

        return self.state, reward, done, self.current_markov_state, self.current_timestep
    
    def update_conditions(self, action=None, event=None, extinction=False):
        """
        Met à jour les conditions de l'environnement.

        :param action: Action prise par l'agent. (None ou "anticipate")
        :param event: Événement se produisant à ce timestep. (None ou "injection")
        """
        if action != None:
            if action == "anticipate":
                self.anticipations.append(ToleranceResponse(self.current_timestep))
            else:
                raise ValueError(f"Action inconnue : {action}")

        if event is not None and not extinction:
            if event == "injection":
                self.injections.append(EthanolInjection(self.current_timestep + 0.5))
            else:
                raise ValueError(f"Événement inconnu : {event}")
        return
    
    def update_state(self):
        """
        Met à jour l'état interne en fonction des conditions de l'environnement.
        """
        # Effets des injections
        anticipation_offset = torch.as_tensor(0, dtype=torch.float32)
        injection_offset = torch.as_tensor(0, dtype=torch.float32)
        for injection in self.injections:
            injection_offset = injection_offset + injection.get_effect(self.current_timestep)
        for anticipation in self.anticipations:
            anticipation_offset = anticipation_offset + anticipation.get_effect(self.current_timestep)
        offset = anticipation_offset + injection_offset
        self.state = self.initial_state.clone() + offset
        


    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.

        :return: Tuple contenant :
        - l'état initial
        - l'état markovien initial
        - le timestep initial
        """
        self.current_timestep = 0
        self.state = self.drive.optimal_state.clone()
        self.drive.update_state(self.state)
        self.current_markov_state = 0
        self.anticipations = []
        self.injections = []
        return self.state, self.current_markov_state, self.current_timestep
    
class EthanolInjection():
    def __init__(self, injection_timestep):
        self.injection_timestep = injection_timestep
    
    def ethanol_curve(self, t):
        '''
        Renvoie l'effet de l'injection (ne pas utiliser dans l'environnement).

        :param t: Temps depuis l'injection

        :return: Effet de l'injection (torch.Tensor)
        '''
        t = t - self.injection_timestep  # Adjust time for injection
        if t <= 0:
            return 0
        values = {0.0: 0, 0.5: 1.05, 1.0: 1.38, 1.5: 1.36, 2.0: 1.26, 2.5: 1.12, 3.0: 1.0, 3.5: 0.90, 4.0: 0.83, 4.5: 0.75, 5.0: 0.68, 5.5: 0.61, 6.0: 0.55, 6.5: 0.49, 7.0: 0.44, 7.5: 0.39, 8.0: 0.34, 8.5: 0.3, 9.0: 0.27, 9.5: 0.23, 10.0: 0.21, 10.5: 0.18, 11.0: 0.16, 11.5: 0.14, 12.0: 0.12, 12.5: 0.1, 13.0: 0.09, 13.5: 0.08, 14.0: 0.07, 14.5: 0.06, 15.0: 0.05, 15.5: 0.05, 16.0: 0.04, 16.5: 0.03, 17.0: 0.03, 17.5: 0.02, 18.0: 0.02, 18.5: 0.02, 19.0: 0.02, 19.5: 0.01, 20.0: 0.01, 20.5: 0.01, 21.0: 0.01, 21.5: 0.01, 22.0: 0.01, 22.5: 0.01, 23.0: 0.01, 23.5: 0.01, 24.0: 0}
        
        mean = torch.zeros(1)
        std = torch.ones(1) * 0.05
        noise = torch.normal(mean, std)
        return torch.as_tensor(-values.get(t, 0)) + noise

        
    def get_effect(self, t):
        """
        Renvoie l'effet de l'injection (à utiliser dans l'environnement).

        :param t: Timestep courant

        :return: Effet de l'injection (torch.Tensor)
        """
        return self.ethanol_curve(t - self.injection_timestep)

class ToleranceResponse():
    def __init__(self, response_timestep):
        self.response_timestep = response_timestep
    
    def tolerance_response_curve(self, t):
        """
        Renvoie les effets de la réponse de tolérance. (ne pas utiliser dans l'environnement)

        :param t: Timestep courant

        :return: Effet de la réponse de tolérance (torch.Tensor)
        """
        t = t - self.response_timestep  # Adjust time for injection
        if t <= 0:
            return 0
        values = {0.0: 0.0, 0.5: 0.3, 1.0: 0.52, 1.5: 0.69, 2.0: 0.82, 2.5: 0.91, 3.0: 0.98, 3.5: 1.04, 4.0: 1.09, 4.5: 0.99, 5.0: 0.9, 5.5: 0.81, 6.0: 0.74, 6.5: 0.67, 7.0: 0.61, 7.5: 0.55, 8.0: 0.5, 8.5: 0.45, 9.0: 0.41, 9.5: 0.37, 10.0: 0.34, 10.5: 0.31, 11.0: 0.28, 11.5: 0.25, 12.0: 0.23, 12.5: 0.21, 13.0: 0.19, 13.5: 0.17, 14.0: 0.15, 14.5: 0.14, 15.0: 0.13, 15.5: 0.12, 16.0: 0.1, 16.5: 0.09, 17.0: 0.09, 17.5: 0.08, 18.0: 0.07, 18.5: 0.06, 19.0: 0.06, 19.5: 0.05, 20.0: 0.05, 20.5: 0.04, 21.0: 0.04, 21.5: 0.04, 22.0: 0.03, 22.5: 0.03, 23.0: 0.03, 23.5: 0.02, 24.0: 0.02}
        mean = torch.zeros(1)
        std = torch.ones(1) * 0.05
        noise = torch.normal(mean, std)
        return torch.as_tensor(values.get(t, 0)) + noise

    def get_effect(self, t):
        """
        Retourne l'effet de la réponse de tolérance. (à utiliser dans l'environnement)

        :param t: Timestep courant

        :return: Effet de la réponse de tolérance (torch.Tensor)
        """
        return self.tolerance_response_curve(t - self.response_timestep)
    

