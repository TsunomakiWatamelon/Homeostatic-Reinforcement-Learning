import torch

class Drive:
    def __init__(self, len_state, optimal_state, state_weights=None, n=1, m=1):
        self.len_state = len_state
        self.internal_state = torch.zeros(len_state)
        self.optimal_state = torch.as_tensor(optimal_state, dtype=torch.float32)
        self.n = n
        self.m = m
        self.state_weights = torch.ones(len_state) if state_weights is None else torch.as_tensor(state_weights, dtype=torch.float32)
    
    def update_state(self, state):
        self.internal_state = torch.as_tensor(state, dtype=torch.float32)

    def get_state(self):
        return self.internal_state
    
    def get_state_len(self):
        return self.len_state
    
    def reset_state(self):
        self.internal_state = torch.zeros(self.len_state)

    def get_optimal_state(self):
        return self.optimal_state
    
    def get_drive(self, state=None):
        if state is None:
            state = self.internal_state
        state = torch.as_tensor(state, dtype=torch.float32)
        differences = torch.abs(self.optimal_state - state) ** self.n
        weighted_sum = torch.sum(self.state_weights * differences)
        drive = weighted_sum ** (1 / self.m)
        return drive
    
    def get_reward(self, new_state):
        current_drive = self.get_drive()
        next_drive = self.get_drive(new_state)
        return current_drive - next_drive

    def __str__(self):
        return f"Internal state: {self.internal_state}"
