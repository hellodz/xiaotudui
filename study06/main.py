import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


train_dataset = torchvision.datasets.CIFAR10(
    root="dataset",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
writer = SummaryWriter("logs")
for i in range(10):
    img,target = train_dataset[i]
    writer.add_image("cifar10",img,i)
    print(train_dataset.classes[target])

writer.close()




