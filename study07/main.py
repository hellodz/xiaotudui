import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_dataset = torchvision.datasets.CIFAR10(
    root="dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

test_dataloader = DataLoader(dataset=test_dataset,batch_size=32,shuffle=True,num_workers=0)
writer = SummaryWriter("logs")
step = 0
for data in test_dataloader:
    imgs,indexs = data
    writer.add_images(tag="imgs",img_tensor=imgs,global_step=step)
    step = step + 1

writer.close()

