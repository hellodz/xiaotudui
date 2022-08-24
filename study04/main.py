from PIL import Image
from torchvision import transforms

#用相对路径 / 不会被转移
img_path = "dataset/train/ants/0013035.jpg"

img = Image.open(img_path)

#__init__
tensor_trans = transforms.ToTensor()
#__call__
img_tensor = tensor_trans(img)

print(img_tensor)