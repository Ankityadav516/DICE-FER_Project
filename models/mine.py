import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, input_dim):
        super(MINE, self).__init__()
        self.norm = nn.LayerNorm(input_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)            # (batch, input_dim)
        xy = self.norm(xy)                       # LayerNorm for stability
        return self.net(xy)                      # (batch, 1)
