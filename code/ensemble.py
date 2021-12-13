import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import datetime

device = torch.device('cuda:0')
torch.cuda.empty_cache()

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# hyper parameter
block1 = [2, 2, 2, 2]
block2 = [3, 6, 12, 8]
batch_size = 16
epochs = 20
growth_rate = 12


# 2. 합성곱 신경망(CNN) 정의하기

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
        self.linear = nn.Linear(inner_channels, num_classes)
        
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

class RenseNet_drop(nn.Module):
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
            nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(), nn.Dropout(p=0.5), 
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

class myensemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(myensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, inputs):
        x1 = self.modelA(inputs)
        x2 = self.modelB(inputs)
        x = x1 + x2
        x = nn.Softmax(dim=1)(x)

        return x



modelA = RenseNet(BasicBlock, block1).to(device)
modelB = RenseNet_drop(BasicBlock, block2).to(device)

params = torch.load('./path/quadruple2_rensenet.pth', map_location='cpu')
modelA.load_state_dict(params)
modelA = modelA.to(device)

params = torch.load('Z:/sjyun/HALFrensenet.pth', map_location='cpu')
modelB.load_state_dict(params)
modelB = modelB.to(device)


#modelA = Net1(BasicBlock, block).to(device)
#modelB = Net1(BasicBlock, block).to(device)
#modelC = Net1(BasicBlock, block).to(device)
#modelA.load_state_dict(torch.load('./path/crop.pth'))
#modelB.load_state_dict(torch.load('./path/flip.pth'))
#modelC.load_state_dict(torch.load('./path/rotate.pth'))

#torch.save(modelA.state_dict(), './cifar_net1.pth')
#torch.save(modelA.state_dict(), './cifar_net2.pth')
#model = myensemble(modelA, modelB)
#model = model.to(device)

model = myensemble(modelA, modelB)
model = model.to(device)



class my_dataset(Dataset):
    def __init__(self, load_dir, transforms=None):
        super().__init__()
        self.transform = transforms
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


# 5. 테스트

testload_dir = 'Z:/sjyun/dataset2013/testz.npz'

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

net.eval()
with torch.no_grad():
    for data1 in testloader:
        images, labels = data1[0].to(device), data1[1].to(device)
        outputs = model(images)
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

print('Accuracy: %.2f %%' % (100 * correct / total))
print('Precision: %.2f %%' % (100 * tp / precision_denom))
print('Sensitivity: %.2f %%' % (100 * tp / recall_denom))
print('Specificity: %.2f %%' % (100 * tn / specifi_denom))

del model, modelA, modelB
torch.cuda.empty_cache()