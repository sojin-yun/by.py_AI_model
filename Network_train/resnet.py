import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

import os
import sys
import time
import datetime

# dataset: heatmap_128_LN2_2013_HU
# trainset: 12446, testset: 1312, validationset: 2476
# [2,2,2,2] epochs: 15: Accuracy: 68 %, Precision: 62 %, Sensitivity: 89 %, Specificity: 47 % 
# [2,2,2,2] epochs: 20: Accuracy: 77 %, Precision: 75 %, Sensitivity: 81 %, Specificity: 73 % 536 479 177 120
# [3,4,6,3] epochs: 20, Accuracy: 80 %, Precision: 80 %, Sensitivity: 79 %, Specificity: 80 % 522 531 125 134
# class 3:7로 나눠보자. Raw data 분류해보기, 보틀넥 넣어서 해보기


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
PATH = './path/resnet2222.pth'

# hyperparameter
block = [2, 2, 2, 2]
# block = [3, 4, 6, 3]
batch_size = 16
show_period = 250
valid_period = 150
learning_rate = 0.001
epochs = 20

trainload_dir = '/home/NAS_mount/sjyun/dataset2013/trainz.npz'
testload_dir = '/home/NAS_mount/sjyun/dataset2013/testz.npz'
valid_dir = '/home/NAS_mount/sjyun/dataset2013/validz.npz'

class my_dataset(Dataset):
    def __init__(self, load_dir, transforms=None):
        super().__init__()
        self.transform = transformss
        img = np.load(load_dir)
        self.ct = img['ct']
        self.label = img['label']
        del img

    def __getitem__(self, index):
        img = self.ct[index]
        target = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.ct)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


print('start loading the trainset')
start = time.time()
trainset = my_dataset(load_dir=trainload_dir, transforms=transform)
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print('time: ', times)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
del trainset

"""
print('start loading the validset')
start = time.time()
validset = my_dataset(load_dir=valid_dir, transforms=transform)
sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print('time: ', times)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0)
del validset
"""


# 2. 합성곱 신경망(CNN) 정의하기

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

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
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=2):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            #nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(p=0.5), 
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _make_layer(self, block, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.pool(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

net = ResNet(BasicBlock, block).to(device)


# 3. 손실 함수와 Optimizer 정의하기
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# 4. 신경망 학습하기

train_loss = []
valid_loss = []
accuracy = []

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
    del data, loss, outputs, inputs, labels
"""
    # validation part
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(validloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
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
        del outputs, inputs, labels
    del data, loss
"""
    # accuracy.append(100 * correct / total)

sec = time.time()-start
times = str(datetime.timedelta(seconds=sec)).split(".")
times = times[0]
print("learning time :", times)
print('Finished Training')

torch.save(net.state_dict(), PATH)


# 5. 테스트
print('start loading the testset')
testset = my_dataset(load_dir=testload_dir, transforms=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
del testset

correct = 0
total = 0

tn = 0 # mat[0][0]
tp = 0 # mat[1][1]
fp = 0 # mat[0][1]
fn = 0 # mat[1][0]

precision_denom = 0
recall_denom = 0
specifi_denom = 0

print('start testing')

with torch.no_grad():
    for data1 in testloader:
        images, labels = data1[0].to(device), data1[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            if labels[i] == 0:
                if predicted[i] == 0:
                    tn += 1
                else:
                    fp += 1
            else:
                if predicted[i] == 0:
                    fn += 1
                else:
                    tp += 1
    del data1, images, labels


precision_denom = tp + fp
recall_denom = tp + fn
specifi_denom = fp + tn

print('tp: {}, tn: {}, fp: {}, fn: {}'.format(tp, tn, fp, fn))

print('Accuracy of the network on the 1312 test images: %d %%' % (
    100 * correct / total))
print('Precision: %.2f %%' % (100 * tp / precision_denom))
print('Sensitivity: %.2f %%' % (100 * tp / recall_denom))
print('Specificity: %.2f %%' % (100 * tn / specifi_denom))