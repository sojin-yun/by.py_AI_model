import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

from myHG import *
from myDataset import my_Dataset

# training : img - HGnet - heatmap - embedding : training
# test : IoU score check

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

NET_NAME = 'hgNet'
batch_size = 2
epochs = 60
learning_rate = 0.001

IMGNAME = f'{NET_NAME}: batch={batch_size} startLR={learning_rate}'
IMGPATH = f'./img/{NET_NAME}_loss.png'
trainload_dir = 'Z:/sjyun/tmpppdata/trainz.npz'
valid_dir = 'Z:/sjyun/tmpppdata/validz.npz'


#################네트워크자리#####################
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        
        self.residual_function = nn.Sequential(
            nn.ReLU(), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False))
        
        self.skip_layer = nn.Sequential(
            nn.ReLU(), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False))

        #if in_channels == out_channels:self.need_skip = False
        #else: self.need_skip = True
    
    def forward(self, x):
        #if self.need_skip: residual = self.skip_layer(x)
        #else:residual = x
        out = self.residual_function(x)
        residual = self.skip_layer(x)
        out += residual
        return out

class HourglassModule(nn.Module):
    def __init__(self, num_feats, num_blocks, num_classes):
        super(HourglassModule, self).__init__()
        self.num_feats = num_feats
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.channels = [256,256,384,384,384,512]
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest') #'bilinear'

    def _conv_layer(self, in_feats, out_feats):
        conv = nn.Conv2d(in_feats, out_feats, kernel_size=1, bias=False)
        return conv

    def forward(self, x):
        downsample = []
        for i in range(self.num_blocks):            
            hg_block = Residual(self.channels[i], self.channels[i+1], stride=2)(x)
            downsample.append(hg_block)
            x = hg_block
        hg_block = Residual(self.channels[-2], self.channels[-1], stride=2)(x)
        x = hg_block
        # upsample
        for i in range(self.num_blocks+1, 1, -1):
            upsample = self.upsample_layer(x)
            upsample = self._conv_layer(upsample, self.channels[i], self.channels[i-1])
            hg_out = downsample[i] + upsample
            x = hg_out
        return hg_out

class HourglassNet(nn.Module):
    def __init__(self, block=Residual, num_stacks=2, num_blocks=4, num_classes=2):
        super(HourglassNet, self).__init__()
        self.in_channels = 64
        self.num_feats = 256
        self.num_stacks = num_stacks
        
        # Initial processing of the image (gpu 사용량이 높아서 HG 들어가기 전에 줄여줘)
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = block(self.in_channels, int(self.num_feats/2))
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer2 = block(int(self.num_feats/2), int(self.num_feats/2))
        self.layer3 = block(int(self.num_feats/2), self.num_feats)
        self.hg_block = HourglassModule(self.num_feats, num_blocks=num_blocks, num_classes=num_classes) 
        self.fc = nn.Conv2d(256, 1, kernel_size=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.hg_block(x)
        x = self.fc(x)
        #x = self.fc(x)
        
        return x

##################################################


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

print('start loading the trainset')
#trainset = my_Dataset(load_dir=trainload_dir, transforms=transform)
trainset = my_Dataset(load_dir=trainload_dir)
#trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=trainset.collate_fn)
del trainset

print('start loading the validset')
#validset = my_Dataset(load_dir=valid_dir, transforms=transform)
validset = my_Dataset(load_dir=valid_dir)
#validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=validset.collate_fn)
del validset

gc.collect()
torch.cuda.empty_cache()

net = HourglassNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

# Move to default device
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)


# Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
# To convert iterations to epochs, divide iterations by the number of iterations per epoch
# The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations


# Epochs
for epoch in range(epochs):
    # One epoch's training
    train(train_loader=trainloader,
            model=net,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch)

    # Save checkpoint
    save_checkpoint(epoch, net, optimizer)



def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
    # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    gc.collect()
    torch.cuda.empty_cache()


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
        #inputs, labels = data[0].to(device), data[1].to(device)
        inputs, boxes, labels = data[0].to(device), data[1].to(device), data[2].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, boxes, labels)
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
PATH = f'./path/{NET_NAME}_{epoch+1}.pth'
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