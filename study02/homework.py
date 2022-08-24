# 需求：获取本地文件的蚂蚁和蜜蜂的图片和标签
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self,root_dir,image_dir,label_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.img_list = os.listdir(self.root_dir + "/" + self.image_dir)


    def __getitem__(self, item):
        img = Image.open(self.root_dir + "/" + self.image_dir + "/" + self.img_list[item])
        # python如何读取txt文件内容
        str = self.root_dir + "/" + self.label_dir + "/" + self.img_list[item]
        # python截取特定字符前的字符
        str = str.split(".")[0]+".txt"
        f = open(str, encoding="utf-8")
        # 输出读取到的数据
        label = f.read()
        # 关闭文件
        f.close()
        return img,label
if __name__ == "__main__":

    root_dir = "my_train_dataset/train"
    image_dir = "ants_image"
    label_dir = "ants_label"
    ants_data = MyData(root_dir,image_dir,label_dir)

    img,label = ants_data[0]
    img.show()
    print(label)