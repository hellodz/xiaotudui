import torch
from PIL import Image
from torchvision import transforms
# from model import Tudui
img_path = "data/1.png"
img = Image.open(img_path)
img = img.convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32))
])
img = transform(img)
print(img.shape)
img = torch.reshape(img,[1,3,32,32])
print(img.shape)

#加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tudui = torch.load("tuidui1.pth").to(device)
tudui.eval()
with torch.no_grad():
    img = img.to(device)
    output = tudui(img)
    print(output)
    print(output.argmax(1))
