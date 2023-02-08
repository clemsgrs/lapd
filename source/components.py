import torch.nn as nn
from typing import Optional


class LossFactory:
    def __init__(
        self,
        loss: Optional[str] = "ce",
    ):

        if loss == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "ordinal":
            self.criterion = nn.MSELoss()

    def get_loss(self):
        return self.criterion
