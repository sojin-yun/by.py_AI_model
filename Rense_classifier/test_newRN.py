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
import pandas as pd

from labeling import my_dataset

testload_dir = 'Z:/sjyun/dataset2013/testz.npz'
XLNAME = 'data.xlsx'
#device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

it = [5,10,15,20,25,30]
LOADNAME = './RNpath/RN_6122416'
block = [6,12,24,16]
batch_size = 16
growth_rate = 6

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

        self.features = nn.Sequential()
        for index in range(len(num_block)-1):
            self.features.add_module("rense_block_layer_{}".format(index), self._make_rense_layers(block, inner_channels, num_block[index]))
            inner_channels += growth_rate * num_block[index]
            out_channels = int(compression * inner_channels)
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        self.features.add_module("rense_block_layer_{}".format(len(num_block)), self._make_rense_layers(block, inner_channels, num_block[-1]))
        inner_channels += growth_rate * num_block[-1]
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
        rense_block = nn.Sequential()
        for index in range(nblocks):
            rense_block.add_module('basic_block_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return rense_block
        
    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.features(output)
        output = self.glob_pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output 

dataxl = []

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

    A = f'%.2f' % (100 * correct / total)
    P = f'%.2f' % (100 * tp / precision_denom)
    R = f'%.2f' % (100 * tp / recall_denom)
    S = f'%.2f' % (100 * tn / specifi_denom)
    dataxl.append([A,P,R,S])

df = pd.DataFrame(dataxl, columns = ['A','P','R','S'])
df.to_excel(XLNAME, index = False)