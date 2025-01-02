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

    def step(self, action):
        """
        Simule une étape dans l'environnement.
        :param action: 0 = aucune réponse, 1 = réponse anticipatoire
        :return: (nouvel état, récompense, terminé, info)
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
        self.update_conditions(action_taken, event)
        self.update_state()

        # Calcul de la récompense (négatif si éloigné du setpoint)
        new_state = self.state.clone()
        reward = self.drive.get_reward(new_state)

        # Avancer dans le temps
        self.current_timestep += 0.5 # 0.5 == 30 minutes == 0.5
        done = self.current_timestep >= self.max_timestep

        return self.state, reward, done, self.current_markov_state
    
    def update_conditions(self, action=None, event=None):
        """
        Met à jour les conditions de l'environnement.
        """
        if action != None:
            if action == "anticipate":
                self.anticipations.append(ToleranceResponse(self.current_timestep))
            else:
                raise ValueError(f"Action inconnue : {action}")

        if event is not None:
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
        offset = torch.as_tensor(0, dtype=torch.float32)
        add = 0
        for injection in self.injections:
            offset += injection.get_effect(self.current_timestep)
        for anticipation in self.anticipations:
            offset += anticipation.get_effect(self.current_timestep)
        self.state = self.initial_state.clone() + offset
        


    def reset(self):
        """
        Réinitialise l'environnement pour un nouvel épisode.
        """
        self.current_timestep = 0
        self.state = self.drive.optimal_state.clone()
        self.drive.update_state(self.state)
        self.current_markov_state = 0
        self.anticipations = []
        self.injections = []
        return self.state, self.current_markov_state
    
class EthanolInjection():
    def __init__(self, injection_timestep):
        self.injection_timestep = torch.as_tensor(injection_timestep, dtype=torch.float32)
    
    def ethanol_curve(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        t_injection = torch.tensor(1, dtype=torch.float32)
        
        if t <= t_injection:
            return torch.tensor(0.0, dtype=torch.float32)
        
        t = t - t_injection  # Adjust time for injection
        a = torch.tensor(-0.15, dtype=torch.float32)
        b = torch.tensor(-0.05, dtype=torch.float32)
        c = torch.tensor(0.5, dtype=torch.float32)
        k = torch.tensor(0.5, dtype=torch.float32)
        t_peak = torch.tensor(2.0, dtype=torch.float32)
        t_decay = torch.tensor(5.0, dtype=torch.float32)

        if t < t_peak:  # Quadratic rise
            return a * t**2
        elif t_peak <= t < t_decay:  # Quadratic fall
            return -b * (t - t_peak)**2 + a * t_peak**2
        else:  # Exponential decay
            return -c * torch.exp(-k * (t - t_decay))

        
    def get_effect(self, t):
        """
        Renvoie l'effet de l'injection
        """
        return self.ethanol_curve(t - self.injection_timestep)

class ToleranceResponse():
    def __init__(self, response_timestep):
        self.response_timestep = torch.as_tensor(response_timestep)
    
    def tolerance_response_curve(self, t):
        t = torch.as_tensor(t, dtype=torch.float32)
        t_start = torch.tensor(0.0, dtype=torch.float32)
        t = t - t_start  # Adjust time for injection
        a = torch.tensor(0.15, dtype=torch.float32)
        b = torch.tensor(0.15, dtype=torch.float32)
        c = torch.tensor(-0.7, dtype=torch.float32)
        k = torch.tensor(0.2, dtype=torch.float32)
        t_peak = torch.tensor(3.0, dtype=torch.float32)
        t_decay = torch.tensor(5.5, dtype=torch.float32)

        if t < t_decay:  # Quadratic response
            return -b * (t - t_peak)**2 + a * t_peak**2
        else:  # Exponential decay
            return -c * torch.exp(-k * (t - t_decay))

    def get_effect(self, t):
        """
        Returns the effect of the tolerance response
        """
        return self.tolerance_response_curve(t - self.response_timestep)
    

