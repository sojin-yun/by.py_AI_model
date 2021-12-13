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
import matplotlib.pyplot as plt
from PIL import Image

from labeling import my_dataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

PATH = './path/batch60_rensenet.pth'
IMAGENAME = './image/batch60_rensenet_loss.png'

# hyperparameter
block = [3,4,6,3]
batch_size = 4
epochs = 15
growth_rate = 32
learning_rate = 0.001
show_period = 350
valid_period = 70

# trainset: 12446, validnset: 2476
trainload_dir = 'Z:/sjyun/dataset2013/trainz.npz'
valid_dir = 'Z:/sjyun/dataset2013/validz.npz'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


class BasicBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, stride=1):
        super().__init__()

        out_channels = growth_rate

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.cat([x, nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))], 1)

class Bottleneck(nn.Module):

    def __init__(self, in_channels, growth_rate, stride=1):
        super().__init__()

        inner_channel = int(growth_rate/2)
        out_channels = growth_rate

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(inner_channel), nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, inner_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channel), nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.cat([x, nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))], 1)

class BigBottleneck(nn.Module):

    def __init__(self, in_channels, growth_rate, stride=1):
        super().__init__()

        inner_channel = 4 * growth_rate
        out_channels = growth_rate

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel), nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
               nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.cat([x, nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))], 1)

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

class RenseNet(nn.Module):
    def __init__(self, block, num_block, growth_rate=12, compression=0.5, num_classes=2):
        super().__init__()

        self.growth_rate = growth_rate
        self.compression = compression
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(1, inner_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.avg_pool = nn.MaxPool2d(3, stride=2)

        self.rense_transit_1 = self._make_rense_transit_layers(block, inner_channels, num_block[0])
        inner_channels = int((inner_channels + num_block[0] * growth_rate) * compression)
        self.rense_transit_2 = self._make_rense_transit_layers(block, inner_channels, num_block[1])
        inner_channels = int((inner_channels + num_block[1] * growth_rate) * compression)
        self.rense_transit_3 = self._make_rense_transit_layers(block, inner_channels, num_block[2])
        inner_channels = int((inner_channels + num_block[2] * growth_rate) * compression)
        self.rense_4 = self._make_rense_layers(block, inner_channels, num_block[3])
        inner_channels = inner_channels + num_block[3] * growth_rate
        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.linear = nn.Linear(inner_channels, num_classes)
        self.linear = nn.Sequential( 
            #nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(), nn.Dropout(p=0.5), 
            nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(),
            nn.Linear(int(inner_channels/2), num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _make_rense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('basic_block_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block
        
    def _make_rense_transit_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('basic_block_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        dense_block.add_module('bn', nn.BatchNorm2d(in_channels))
        out_channels = int(self.compression * in_channels)
        dense_block.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        dense_block.add_module('pool', nn.AvgPool2d(2, stride=2))
        return dense_block

    def forward(self, x):
        output = self.conv1(x)
        output = self.avg_pool(output)
        output = self.rense_transit_1(output)
        output = self.rense_transit_2(output)
        output = self.rense_transit_3(output)
        output = self.rense_4(output)
        output = self.glob_pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output 


print('start loading the trainset')
trainset = my_dataset(load_dir=trainload_dir, transforms=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
del trainset

print('start loading the validset')
validset = my_dataset(load_dir=valid_dir, transforms=transform)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0)
del validset

net = RenseNet(BasicBlock, block, growth_rate=growth_rate).to(device)

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
    net.train()
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

    # validation part
    correct = 0
    total = 0
    running_loss = 0.0
    net.eval()
    for i, data in enumerate(validloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        if i % valid_period == valid_period - 1:
            print('[%d, %5d] validation loss: %.3f' % (epoch + 1, i + 1, running_loss / valid_period))
            valid_loss.append(running_loss/valid_period)
            running_loss = 0.0
        del inputs, labels, outputs            
        gc.collect()
        torch.cuda.empty_cache()
    del data, loss
    gc.collect()
    torch.cuda.empty_cache()

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print("learning time :", times)
print('Finished Training')

torch.save(net.state_dict(), PATH)

x1 = range(len(train_loss))
x2 = range(len(valid_loss))

plt.plot(x1, train_loss, 'b', label = 'Training loss')
plt.plot(x2, valid_loss, 'r', label = 'Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('RenseNet')
plt.legend(loc='upper right')
plt.savefig(IMAGENAME)