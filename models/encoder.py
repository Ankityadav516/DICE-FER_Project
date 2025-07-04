import torch
import torch.nn as nn
import torchvision.models as models

class ExpressionEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(ExpressionEncoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

        #  Init
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.bn(self.fc(x))

class IdentityEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(IdentityEncoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

        #  Init
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.bn(self.fc(x))
