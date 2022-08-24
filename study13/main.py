import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, Flatten

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



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
            Flatten()
            , Linear(1024, 64)
            ,Linear(64, 10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

model = Tudui()
#检测网络是否正确
input = torch.ones(size=(64,3,32,32))
output = model(input)
print(output.shape)
