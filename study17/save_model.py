import torch
from model import Tudui


tudui = Tudui()
#方式1 保存模型结构+模型参数
torch.save(tudui,"model1.pth")

#方式2 保存模型参数
torch.save(tudui.state_dict(),"model2.pth")