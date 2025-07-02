import torch
import torch.nn as nn
import torchvision.models as models

class ExpressionEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(ExpressionEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.fc(x)

class IdentityEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(IdentityEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.fc(x)
    