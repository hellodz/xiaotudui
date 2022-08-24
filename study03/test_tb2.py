from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter(log_dir="logs")

img_path = "D:\\project\\python\\xiaotudui\\study03\\dataset\\train\\ants\\2278278459_6b99605e50.jpg"
img = Image.open(img_path)
print(type(img))
# PIL 转 numpy
np_img = np.array(img)
print(np_img.shape)
writer.add_image(tag="img",img_tensor=np_img,global_step=2,dataformats="HWC")

writer.close()
