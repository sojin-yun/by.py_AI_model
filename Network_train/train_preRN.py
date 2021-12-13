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
from densenet import Bottleneck, DenseNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

PATH = './path/densenet.pth'

# hyperparameter
block = [6, 12, 24, 16]
learning_rate = 0.001
batch_size = 4
epochs = 10
show_period = 250

trainload_dir = 'Z:/sjyun/dataset2013/trainz.npz'
testload_dir = 'Z:/sjyun/dataset2013/testz.npz'
valid_dir = 'Z:/sjyun/dataset2013/validz.npz'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])


print('start loading the trainset')
trainset = my_dataset(load_dir=trainload_dir, transforms=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
del trainset

net = DenseNet(Bottleneck, block).to(device)

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