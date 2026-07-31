import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Deep Q-Network for CartPole Environment."""
    
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
