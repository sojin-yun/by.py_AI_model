import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)
b = b+1
print(a)
print(b)
a = torch.from_numpy(b)
print(a)
"""



a = torch.ones(5,3,3)
a[0] = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

"""
for i in range(5):
    for j in range(3):
        for k in range(3-1,0,-1):
            if a[i][j][k]>a[i][j][k-1]:
                a[i][j][k-1] = a[i][j][k]
"""

class Left_Pool(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def forward(self,x):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                for k in range(self.x.size(dim=2)-1, 0, -1):
                    if x[i][j][k] > x[i][j][k-1]:
                        x[i][j][k-1] = x[i][j][k]
        return x

class Top_Left_Corner(nn.Module):
    def __init__(self,x):
        super(Top_Left_Corner, self).__init__()
        self.left_pool = Left_Pool()
        self.x = x
    def forward(self, x):
        left = self.left_pool(self.x)
        out = left
        return out

b = Top_Left_Corner(a)
b.forward(a)
print(a)

class Top_Left_Corner(nn.Module):
    def __init__(self, in_channels, img_size, x):
        super(Top_Left_Corner, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.x = x
        self.left_pool = Left_Pool()
        
    def _left_pool(self, ):
        for i in range(self.x.size(dim=0)):
            for j in range(self.x.size(dim=1)):
                for k in range(self.x.size(dim=2)-1, 0, -1):
                    if self.x[i][j][k] > self.x[i][j][k-1]:
                        self.x[i][j][k-1] = self.x[i][j][k]
        return self.x

    def forward(self, x):
        top = self.top_pool(x)
        left = self.left_pool(x)
        out = top + left
        return out

        