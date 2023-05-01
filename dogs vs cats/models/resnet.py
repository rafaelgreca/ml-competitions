import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = resnet18(weights="ResNet18_Weights.DEFAULT")

        for param in self.model.parameters():
            param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.model(X)
        return X
