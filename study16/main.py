import torch
import torchvision
from torch import nn, optim
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential, Flatten

from torch.utils.data import DataLoader
from torchvision import transforms

vgg16 = torchvision.models.vgg16()
print(vgg16)

# vgg16.classifier.add_module("7",Linear(1000,10))
vgg16.classifier[6] = Linear(1000,10)
print(vgg16)


