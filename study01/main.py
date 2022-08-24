#导入torch
import torch
#测试cuda是否可用 Returns a bool indicating if CUDA is currently available.
print(torch.cuda.is_available())
#打开工具箱-dir
print(dir(torch))
print(dir(torch.cuda))
#输入和上面的不一样 是__XXX__
print(dir(torch.cuda.is_available))
#查看功能-help 注意：不要加括号 is_available
print(help(torch.cuda.is_available))