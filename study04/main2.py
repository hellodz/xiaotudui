# 将np转tensor，用tensorboard显示
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2
#用相对路径 / 不会被转移
img_path = "dataset/train/ants/0013035.jpg"

#无法提示
cv_img = cv2.imread(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(cv_img)

#review
writer = SummaryWriter("logs")
writer.add_image(tag="tensor_img",img_tensor=tensor_img,global_step=1)
writer.close()