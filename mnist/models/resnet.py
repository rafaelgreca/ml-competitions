import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        # for param in self.model.parameters():
        #     param.requires_grad = False

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.model(X)
        return X
