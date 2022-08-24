#使用gpu训练
import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# 引入神经网络
from model import Tudui
from torch import nn, optim
from tqdm import tqdm
# 准备训练和测试数据集
train_dataset = torchvision.datasets.CIFAR10(
    root="dataset",
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_dataset = torchvision.datasets.CIFAR10(
    root="dataset",
    train=False,
    transform=transforms.ToTensor(),
    download=True)
# lenth 长度
train_data_len = len(train_dataset)
test_data_len = len(test_dataset)
print("训练数据集的长度：{len}".format(len=train_data_len))#string格式化输出
print("测试数据集的长度：{len}".format(len=test_data_len))
# 加载训练和测试数据集
train_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 创建网络模型
tudui = Tudui()
tudui = tudui.to(device)
# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 定义学习速率和优化器
learning_rate = 0.01
optim = torch.optim.SGD(tudui.parameters(),learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数(累计每一批次，不是所有图片）
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 15

writer = SummaryWriter("logs")
# 训练数据
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i+1))

    # 训练一轮数据
    tudui.train()
    for data in tqdm(train_dataloader):
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = tudui(imgs)
        # 计算损失
        loss = loss_fn(output,targets)
        # 优化器调优
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step += 1
        # 避免输出太多
        if total_train_step % 100 ==0:
            print("训练次数：{}，Loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_acc = 0
    with torch.no_grad(): #避免调优

        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = tudui(imgs)
            outputs = tudui(imgs)
            # 计算损失
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            # argmax为1表示，方向是横向比较
            acc = (outputs.argmax(1)==targets).sum()
            total_acc += acc

        print("整体测试集上Loss：{}".format(total_test_loss))
        print("整体测试集上正确率：{}".format(total_acc/test_data_len))

        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_acc", total_acc, total_test_step)
        total_test_step += 1
    # 保存模型
    # torch.save(tudui,"tuidui_{}.pth".format(i))
torch.save(tudui,"tuidui15.pth")

writer.close()
