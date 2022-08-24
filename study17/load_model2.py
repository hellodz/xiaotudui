import torch
#方式2需要这一行，不然不行
from model import Tudui



#方式2 加载模型参数
data = torch.load("model2.pth")
print(data)
tudui2= Tudui()
tudui2.load_state_dict(data)
print(tudui2)

