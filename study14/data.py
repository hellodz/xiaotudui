import torch
from torch import nn

loss = nn.L1Loss()
input = torch.tensor([1,2,3], dtype=torch.float32)
target = torch.tensor([1,2,5] ,dtype=torch.float32)
output = loss(input, target)
print(output)
# ((1-1)+(2-2)+(3-5))/3=-0.66666667

loss = nn.MSELoss()
input = torch.tensor([1,2,3], dtype=torch.float32)
target = torch.tensor([1,2,5] ,dtype=torch.float32)
output = loss(input, target)
print(output)

loss = nn.CrossEntropyLoss()
x = torch.tensor([0.1,0.2,0.3])
print(x.shape)
# 一张图片 N C 说一下自己的理解，但不是很确定
x = torch.reshape(x,(1,3))
print(x.shape)
# 一张图片的结果 N
y = torch.tensor([1])
print(y.shape)
output = loss(x, y)
print(output)





