#Ctrl+SummaryWriter 创建一个tensorboard文件, 文件保存在log_dir目录写文件，该文件可以被tensorboard解析
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs")

# writer.add_image()
for i in range(100):
    writer.add_scalar("y=3x",3*i,i)

writer.close()
