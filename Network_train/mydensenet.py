import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import gc
import sys
import time
import datetime
import numpy as np

from labeling import my_dataset


## [6,12,24,16]

# batch:16, epoch:20
# tp: 489, tn: 520, fp: 136, fn: 167
# Accuracy: 76 % Precision: 78.24 % Sensitivity: 74.54 % Specificity: 79.27 %

# batch:8, epoch:20, growth_rate=32
# learning time : 4:15:37
# tp: 490, tn: 517, fp: 139, fn: 166
# Accuracy: 76 % Precision: 77.90 % Sensitivity: 74.70 % Specificity: 78.81 %

# batch:16, epoch:20, dropout 추가
# tp: 575, tn: 476, fp: 180, fn: 81
# Accuracy : 80 % Precision: 76.16 % Sensitivity: 87.65 % Specificity: 72.56 %


## [6,12,32,32] 

# batch:16, epoch:20
# learning time : 2:14:54
# tp: 430, tn: 550, fp: 106, fn: 226
# Accuracy: 74 % Precision: 80.22 % Sensitivity: 65.55 % Specificity: 83.84 %

## [6,12,36,24]

# batch:16, epoch:20


# batch:16, epoch:20, growth_rate=32
# learning time : 8:54:11
# tp: 480, tn: 539, fp: 117, fn: 176
# Accuracy: 77 % Precision: 80.40 % Sensitivity: 73.17 % Specificity: 82.16 %


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

PATH = './path/mydensenet.pth'

# hyperparameter
block = [6, 12, 24, 16]
learning_rate = 0.001
batch_size = 16
growth_rate = 12
epochs = 20
show_period = 250

trainload_dir = 'Z:/sjyun/dataset2013/trainz.npz'
testload_dir = 'Z:/sjyun/dataset2013/testz.npz'
valid_dir = 'Z:/sjyun/dataset2013/validz.npz'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        inner_channel = 4 * growth_rate

        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

class MyDenseNet(nn.Module):
    def __init__(self, block, num_block, growth_rate=12, compression=0.5, num_classes=2):
        super().__init__()

        self.growth_rate = growth_rate
        self.compression = compression
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(1, inner_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.avg_pool = nn.MaxPool2d(3, stride=2)

        self.dense_transit_1 = self._make_dense_transit_layers(block, inner_channels, num_block[0])
        inner_channels = int((inner_channels + num_block[0] * growth_rate) * compression)
        self.dense_transit_2 = self._make_dense_transit_layers(block, inner_channels, num_block[1])
        inner_channels = int((inner_channels + num_block[1] * growth_rate) * compression)
        self.dense_transit_3 = self._make_dense_transit_layers(block, inner_channels, num_block[2])
        inner_channels = int((inner_channels + num_block[2] * growth_rate) * compression)
        self.dense_4 = self._make_dense_layers(block, inner_channels, num_block[3])
        inner_channels = inner_channels + num_block[3] * growth_rate
        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Sequential(
            nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(int(inner_channels/2), num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

    def _make_dense_transit_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        dense_block.add_module('bn', nn.BatchNorm2d(in_channels))
        out_channels = int(self.compression * in_channels)
        dense_block.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        dense_block.add_module('pool', nn.AvgPool2d(2, stride=2))
        return dense_block

    def forward(self, x):
        output = self.conv1(x)
        output = self.avg_pool(output)
        output = self.dense_transit_1(output)
        output = self.dense_transit_2(output)
        output = self.dense_transit_3(output)
        output = self.dense_4(output)
        output = self.glob_pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output 


print('start loading the trainset')
trainset = my_dataset(load_dir=trainload_dir, transforms=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
del trainset

net = MyDenseNet(Bottleneck, block, growth_rate=growth_rate).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

train_loss = []
valid_loss = []
accuracy = []

gc.collect()
torch.cuda.empty_cache()

print('start learning')
start = time.time()

for epoch in range(epochs): 
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 

        running_loss += loss.item() 
        if i % show_period == show_period - 1: 
            print('[%d, %5d] train loss: %.3f' % (epoch + 1, i + 1, running_loss / show_period))
            train_loss.append(running_loss/show_period)
            running_loss = 0.0

        del inputs, labels, outputs            
        gc.collect()
        torch.cuda.empty_cache()


sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print("learning time :", times)
print('Finished Training')

torch.save(net.state_dict(), PATH)