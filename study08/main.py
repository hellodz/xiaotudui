import torch
from torch import nn


class Tudui(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    #备注：ward 没有generate
    def forward(self,input):
        output = input + 1
        return output

if __name__ == "__main__":
    model = Tudui()
    # help
    x = torch.tensor([0,1])
    y = model(x)
    print(y)