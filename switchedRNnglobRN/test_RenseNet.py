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

testload_dir = 'Z:/sjyun/dataset2013/testz.npz'
#device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

it = [5,10,15,20,25]
LOADNAME = './RNpath/globRN'
block = [4,4,4]
batch_size = 32
growth_rate = 12
learning_rate = 0.001

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
    

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

print('start loading the testset')
testset = my_dataset(load_dir=testload_dir, transforms=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
del testset

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        out_channels = growth_rate
        self.BRC_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
    def forward(self, x):
        return torch.cat([x, self.BRC_layer(x)], 1)

class RenseBlock(nn.Module):
    def __init__(self, block, in_channels, nblocks, growth_rate):
        super().__init__()
        
        inner_channel = in_channels
        out_channels = in_channels + nblocks * growth_rate

        self.rense_block = nn.Sequential()
        for index in range(nblocks):
            self.rense_block.add_module('basic_block_layer_{}'.format(index), block(inner_channel, growth_rate))
            inner_channel += growth_rate

        self.skip_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, x):
        return self.rense_block(x) + self.skip_connection(x)

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
    def __init__(self, block, nblocks, growth_rate=12, compression=0.5, num_classes=2):
        """
        block: BasicBlock, Bottleneck
        num_block: the number of basicblks(or bottleN) in each renseblk (renseblk은 우선 3으로, 4로 할 지는 추후에 생각)
        """
        super().__init__()

        self.growth_rate = growth_rate
        self.compression = compression
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(1, inner_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.avg_pool = nn.AvgPool2d(3, stride=2)
        #self.avg_pool = nn.MaxPool2d(3, stride=2)
        self.features = nn.Sequential()
        for index in range(len(nblocks)-1):
            self.features.add_module("rense_block_layer_{}".format(index), RenseBlock(block, inner_channels, nblocks[index], growth_rate))
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(compression * inner_channels)
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        self.features.add_module("rense_block_layer_{}".format(len(nblocks)), RenseBlock(block, inner_channels, nblocks[-1], growth_rate))
        inner_channels += growth_rate * nblocks[-1]
        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.linear = nn.Linear(inner_channels, num_classes)
        self.linear = nn.Sequential( 
            nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(),
            nn.Linear(int(inner_channels/2), num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        output = self.conv1(x)
        output = self.avg_pool(output)
        output = self.features(output)
        output = self.glob_pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output 

for iteration in it:
    LOADPATH = f'{LOADNAME}_{iteration}.pth'
    net = RenseNet(BasicBlock, block, growth_rate=growth_rate).to(device)
    net.load_state_dict(torch.load(LOADPATH), strict=False)

    correct = 0
    total = 0

    tn = 0
    tp = 0
    fp = 0
    fn = 0 

    precision_denom = 0
    recall_denom = 0
    specifi_denom = 0

    print('start testing')

    net.eval()
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

    print("============================================")
    print(f'iteration: {iteration}')
    print('tp: {}, tn: {}, fp: {}, fn: {}'.format(tp, tn, fp, fn))
    print('Accuracy:    %.2f %%' % (100 * correct / total))
    print('Precision:   %.2f %%' % (100 * tp / precision_denom))
    print('Sensitivity: %.2f %%' % (100 * tp / recall_denom))
    print('Specificity: %.2f %%' % (100 * tn / specifi_denom))  
    print("============================================")