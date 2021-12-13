import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import os
import gc
import sys
import time
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

from labeling import my_dataset

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
#torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
#torch.backends.cudnn.deterministic = True # 속도 느려지니 연구 후반 단계에 사용
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

NET_NAME = 'RN'
block = [3,4,6,3]
batch_size = 16
growth_rate = 32
epochs = 100
learning_rate = 0.001

IMGNAME = f'{NET_NAME}: block={block} batch={batch_size} gr={growth_rate} startLR={learning_rate}'
IMGPATH = f'./RNimg/{NET_NAME}_loss.png'
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
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return torch.cat([x, nn.ReLU()(self.residual_function(x) + self.shortcut(x))], 1)

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
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.relu1 = nn.ReLU(inplace=True)
        #self.avg_pool = nn.MaxPool2d(3, stride=2)

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
            nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(), 
            #nn.Dropout(p=0.2)
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
        output = self.bn1(output)
        output = self.relu1(output)
        #output = self.avg_pool(output)
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

gc.collect()
torch.cuda.empty_cache()

net = RenseNet(BasicBlock, block, growth_rate=growth_rate).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

print('start learning\n')
start = time.time()

for epoch in range(epochs): 
    print("epoch: ", epoch + 1)
    print("training [", end='')
    total = 0
    correct = 0
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        if i % int(len(trainloader)/40) == 0:
            print("#", end='')
    print("]")
    del inputs, labels, outputs   
    gc.collect()
    torch.cuda.empty_cache()

    # validation part
    val_total = 0
    val_correct = 0
    val_running_loss = 0.0
    print("validation [", end='')
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(validloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_running_loss += loss.item()
            if i % int(len(validloader)/38) == 0:
                print("#", end='')
        print("]")
        del inputs, labels, outputs            
        gc.collect()
        torch.cuda.empty_cache()
    scheduler.step()

    # Save the network every 5 epochs
    if (epoch+1) % 5 == 0:
        torch.save(net.state_dict(), f"./RNpath/{NET_NAME}_{epoch+1}.pth")

    # Print loss
    train_loss.append(running_loss/len(trainloader))
    valid_loss.append(val_running_loss/len(validloader))
    train_acc.append(100 * correct / total)
    valid_acc.append(100 * val_correct / val_total)
    print("training loss:   {:.3f}".format(train_loss[-1]))
    print("validation loss: {:.3f}".format(valid_loss[-1]))
    print("training accuracy:   {:.2f} %".format(train_acc[-1]))
    print("validation accuracy: {:.2f} %".format(valid_acc[-1]))
    print("====================================================")


sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print("learning time :", times)
print('Finished Training')

# Save the last one
PATH = f'./RNpath/{NET_NAME}_{epoch+1}.pth'
if os.path.exists(PATH):
    print('The last network is already saved.')
else:
    torch.save(net.state_dict(), PATH)

x1 = range(epochs)

plt.subplot(1,2,1), plt.grid(), plt.tick_params(labelsize=8)
plt.plot(x1, train_loss, 'b', label = 'Training loss')
plt.plot(x1, valid_loss, 'r', label = 'Validation loss')
plt.xlabel('epoch'), plt.title('Loss', size=10)
plt.legend(loc='upper right', fontsize=8)

plt.subplot(1,2,2), plt.grid(), plt.tick_params(labelsize=8)
plt.plot(x1, train_acc, 'b', label = 'Training accuracy')
plt.plot(x1, valid_acc, 'r', label = 'Validation accuracy')
plt.xlabel('epoch'), plt.title('Accuracy', size=10)
plt.legend(loc='upper right', fontsize=8)

plt.suptitle(IMGNAME)
plt.tight_layout()

plt.savefig(IMGPATH)

"""

class Bottleneck(nn.Module):

    def __init__(self, in_channels, growth_rate, stride=1):
        super().__init__()

        inner_channels = int(growth_rate/2)
        out_channels = growth_rate

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(inner_channels), nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels), nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, bias=False),
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
"""