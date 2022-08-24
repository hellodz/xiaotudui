from PIL import Image
from torchvision import transforms

#将一张图片转tensor
img_path="dataset/train/ants/0013035.jpg"
img=Image.open(img_path)
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
#将图片归一化
print(img_tensor[2][2][3])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[2][2][3])




