# 数据集CIFAR10，每次64张图片加载
# 定义CNN 最大池化（3）
# 通过writer显示卷积前后的图片
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d

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
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self,x):
        x = self.maxpool1(x)
        return x

model = Tudui()
writer = SummaryWriter("logs")
step = 0
for data in test_dataloader:
    imgs,_ = data
    ouput = model(imgs)
    writer.add_images("input",imgs,step)
    # ouput = torch.reshape(ouput,[-1,3,30,30])
    writer.add_images("output",ouput,step)

    step = step + 1
writer.close()