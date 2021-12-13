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
import gc
import sys
import time
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device) 

PATH = './vgg_pth/vgg.pth'
IMAGENAME = './vgg_img/vgg_loss.png'

batch_size = 4
epochs = 20
learning_rate = 0.005
show_period = 350
valid_period = 70

trainload_dir = 'Z:/sjyun/dataset2013/trainz.npz'
valid_dir = 'Z:/sjyun/dataset2013/validz.npz'

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

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


print('start loading the trainset')
trainset = my_dataset(load_dir=trainload_dir, transforms=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
del trainset

print('start loading the validset')
validset = my_dataset(load_dir=valid_dir, transforms=transform)
validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0)
del validset

class Net(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.convlayer = nn.Sequential(
            
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), # NEW

            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), #add
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), #add
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), #add
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # NEW
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fclayer = nn.Sequential(
            nn.Linear(32768, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 2048), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(2048, 500), nn.ReLU(), 
            nn.Linear(500, 2)
        )

    def forward(self, x):
        x = self.convlayer(x)
        x = torch.flatten(x, 1) 
        x = self.fclayer(x)
        return x

net = Net().to(device)


criterion = nn.CrossEntropyLoss().to(device)
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
    with torch.no_grad():
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
plt.title('RenseNet_dropout')
plt.legend(loc='upper right')
plt.savefig(IMAGENAME)

# 5. 테스트
print('start loading the testset')
testset = my_dataset(load_dir=testload_dir, transforms=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
del testset

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

print('tp: {}, tn: {}, fp: {}, fn: {}'.format(tp, tn, fp, fn))

print('Accuracy of the network on the 1312 test images: %.2f %%' % (
    100 * correct / total))
print('Precision: %.2f %%' % (100 * tp / precision_denom))
print('Sensitivity: %.2f %%' % (100 * tp / recall_denom))
print('Specificity: %.2f %%' % (100 * tn / specifi_denom))