import torch
import torchvision
from torch import nn, optim
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, Flatten

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x = self.model(x)
        return x



