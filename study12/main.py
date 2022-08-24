# 数据集CIFAR10，每次64张图片加载
# 定义CNN 最大池化（3）
# 通过writer显示卷积前后的图片
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_dataset = torchvision.datasets.CIFAR10(
    root="dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)

class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.line1 = Linear(196608,10)

    def forward(self,x):
        x = self.line1(x)
        return x

model = Tudui()
for data in test_dataloader:
    imgs,_ = data
    # input = torch.reshape(imgs,[1,1,1,-1])
    input = torch.flatten(imgs)
    ouput = model(input)
    print(input.shape)
    print(ouput.shape)
    break