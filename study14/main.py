import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, Flatten

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


test_dataset = torchvision.datasets.CIFAR10(
    root="dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=True)

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

loss = nn.CrossEntropyLoss()
model = Tudui()
for data in test_dataloader:
    input,targets = data
    ouput = model(input)
    result_loss = loss(ouput,targets)
    # 1
    print(result_loss)
    result_loss.backward()
    break

