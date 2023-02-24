import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from omegaconf import DictConfig

from source.model_utils import Attn_Net_Gated


class ModelFactory:
    def __init__(
        self,
        arch: str,
        in_channels: int,
        num_classes: int = 2,
        model_options: Optional[DictConfig] = None,
    ):

        if model_options.agg_method == "max_slide":
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
        elif model_options.agg_method == "self_att":
            if arch == "cnn":
                self.model = SelfAttConvNet(
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

        logits = self.fc(x)

        return logits


class SelfAttConvNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, hidden_dim: int = 192, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.act = nn.ReLU()
        self.flat = nn.Flatten()

        self.global_phi = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=3,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=hidden_dim, D=hidden_dim, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):

        # x = [ [M1, 192], [M2, 192], ...]
        N = len(x)
        slide_seq = []
        for n in range(N):
            y = x[n]
            y = self.conv1(y)
            y = self.act(y)
            y = self.conv2(y)
            y = self.act(y)
            y = self.pool(y)
            y = self.flat(y)
            slide_seq.append(y)

        slide_seq = torch.cat(slide_seq, dim=0)
        # slide_seq = [N, 192]
        z = self.global_phi(slide_seq)
        z = self.global_transformer(z.unsqueeze(1)).squeeze(1)
        att, z = self.global_attn_pool(z)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        z_att = torch.mm(att, z)
        z_p = self.global_rho(z_att)

        logits = self.fc(z_p)
        
        return logits


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