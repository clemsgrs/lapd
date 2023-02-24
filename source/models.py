import timm
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from omegaconf import DictConfig


class ModelFactory:
    def __init__(
        self,
        arch: str,
        in_channels: int,
        num_classes: int = 2,
        model_options: Optional[DictConfig] = None,
    ):

        if arch in timm.list_models('*'):
            self.model = timm.create_model(
                arch,
                pretrained=False,
                num_classes=num_classes,
                in_chans=in_channels,
            )
        elif arch == "cnn":
            self.model = BasicConvNet(
                in_channels=in_channels,
                num_classes=num_classes,
            )

    def get_model(self):
        return self.model


class BasicConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.act = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.act(x)

        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        
        return x


class SimpleConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.act = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.act(x)
        
        x = self.conv3(x)
        x = self.act(x)

        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        
        return x