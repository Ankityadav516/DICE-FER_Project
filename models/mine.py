import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class MINE(nn.Module):
    def __init__(self, input_dim):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 512)),
            nn.ReLU(),
            nn.Dropout(0.2),
            spectral_norm(nn.Linear(512, 256)),
            nn.ReLU(),
            nn.Dropout(0.2),
            spectral_norm(nn.Linear(256, 1))
        )

    def forward(self, x, y):
        joint = torch.cat([x, y], dim=1)
        return self.net(joint).squeeze()
