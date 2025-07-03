import torch
import torch.nn as nn

class MINE(nn.Module):
    def __init__(self, input_dim):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 512),  # 128 (z1) + 128 (z2) = 256
            nn.ReLU(),
            nn.Linear(512, 1)
        )


    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))
