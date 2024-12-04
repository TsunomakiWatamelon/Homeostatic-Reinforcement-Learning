class HomeostaticEnvironment:
    def __init__(self, H, setpoints, weights, exponents, effects):
        """
        Initialise l'environnement homéostatique.
        """
        self.state = H 
        self.setpoints = setpoints
        self.weights = weights
        self.exponents = exponents
        self.effects = effects # Effects peut être un dict qui contients les résultats des actions a faire (comme par exemple manger, ou se secouer)

    def update_state(self, action):
        """
        Met à jour l'état interne en fonction de l'action prise.
        """
        if action in self.effects:
            effect = self.effects[action]
            self.state = [self.state[i] + effect[i] for i in range(len(self.state))]
        else:
            print("Error : Action does not exist :", action)

    def step(self, action):
        # Mise a jour de l'etats interne
        self.state = self.update_state(self.state, action, self.effects) 

        # Calcule de la reward
        reward =  # TODO : ici ajouter la fonction de reward codé sur le dossier models
        return self.state, reward
