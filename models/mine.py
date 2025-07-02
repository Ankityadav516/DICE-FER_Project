class MINE(nn.Module):
    def __init__(self, input_dim):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))