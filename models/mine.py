import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, input_dim):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        # Normalize inputs for stability
        xy = (xy - xy.mean(dim=0)) / (xy.std(dim=0) + 1e-8)
        return self.net(xy)
