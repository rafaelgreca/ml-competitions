import torch
import torch.nn as nn


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        pool_size: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.block(X)
        return X


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features), nn.ReLU()
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.block(X)
        return X


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=2, stride=1, padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=12544, out_features=10),
        )
        self.model.apply(init_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.model(X)
        return X


class DeepCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(
                in_channels=1, out_channels=64, kernel_size=2, stride=1, pool_size=2
            ),
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, pool_size=2
            ),
            ConvBlock(
                in_channels=128, out_channels=128, kernel_size=4, stride=1, pool_size=2
            ),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            LinearBlock(in_features=128, out_features=64),
            nn.Dropout(p=0.5),
            LinearBlock(in_features=64, out_features=10),
        )
        self.model.apply(init_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.model(X)
        return X
