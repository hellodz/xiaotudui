#导入Dataset 后面就可以直接用Dataset
from torch.utils.data import Dataset
#读取图片
#PIL：Python Imaging Library，Python图像处理标准库
from PIL import Image
#对文件夹或文件进行操作，但特别让人容易联想操作系统，就感觉很麻烦
import os

#help 说的继承
class MyData(Dataset):
    #为整个class提供一个全局变量,特别像java的构造函数
    # 注意不是int,是init
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        #图片集
        #os.listdir 返回一个包含目录中文件名的列表
        self.img_list = os.listdir(self.root_dir + "/" + self.label_dir)

    # 获得其中每一个图片
    def __getitem__(self, item):
        # 备注 self. 不要忘记
        img = Image.open(self.root_dir + "/" + self.label_dir + "/" + self.img_list[item])
        label = self.label_dir
        return img,label
    def __len__(self):
        #len(列表)
        return len(self.img_list)

#主函数 不要写成mian
if __name__ == "__main__":
    # 需求：获取本地文件的蚂蚁和蜜蜂的图片和标签
    # 输入本地图片路径
    #加双斜杆转义 习惯上等号两边都空格
    root_dir = "dataset/train"

    ants_data = MyData(root_dir,"ants")
    bees_data = MyData(root_dir,"bees")

    img,label = ants_data[0]
    img.show()
    # Dataset 可以直接相加
    train_data = ants_data + bees_data
    print(len(train_data))


