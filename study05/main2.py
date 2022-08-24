# 使用Compose转tensor再将图片resize成(512,512)
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

#将一张图片转tensor
img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)
trans_compose = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor()
])
img_tensor = trans_compose(img)

writer = SummaryWriter("logs")
writer.add_image(tag="tensor",img_tensor=img_tensor)
writer.close()