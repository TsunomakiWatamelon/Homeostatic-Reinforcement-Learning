# Drive module for the homeostatic reinforcement learning model

import torch

class Drive():
    def __init__(self, len_state, optimal_state, state_weights=None, n=1, m=1):
        self.len_state = len_state
        self.internal_state = torch.zeros(len_state)
        self.optimal_state = optimal_state
        self.n = n
        self.m = m

        if state_weights is None:
            self.state_weights = torch.ones(len_state)
    
    def update_state(self, state):
        self.internal_state = state

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
        differences = torch.abs(self.optimal_state - state) ** self.n
        weighted_sum = torch.sum(self.state_weights * differences)
        return weighted_sum ** (1 / self.m)
    
    def get_reward(self, new_state):
        current_drive = self.get_drive()
        next_drive = self.get_drive(new_state)

        return current_drive - next_drive

    def __str__(self):
        return f"Internal state: {self.internal_state}"