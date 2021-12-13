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
#import datetime

from myHG import *
from myDataset import my_Dataset

# Data parameters
trainload_dir = 'Z:/sjyun/tmpppdata/trainz.npz'
valid_dir = 'Z:/sjyun/tmpppdata/validz.npz'

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = 2  # number of different types of objects
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(device)

# Learning parameters
batch_size = 2  # batch size
iterations = 120000  # number of iterations to train
workers = 0  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay

NET_NAME = 'hgNet'
IMGNAME = f'{NET_NAME}: batch={batch_size} startLR={lr}'
IMGPATH = f'./img/{NET_NAME}_loss.png'

def main():
    """
    Training.
    """
    global decay_lr_at

    # Initialize model or load checkpoint

    start_epoch = 0
    model = HourglassNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Custom dataloaders
    trainset = my_Dataset(load_dir=trainload_dir, transforms=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                                shuffle=True, num_workers=0)  # note that we're passing the collate function here
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
    #                                           num_workers=0, collate_fn=trainset.collate_fn,
    #                                           pin_memory=True) 
    
    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(trainset) // 2)
    decay_lr_at = [it // (len(trainset) // 2) for it in decay_lr_at]
    print(epochs)

    # Epochs
    for epoch in range(start_epoch, epochs):
        print("epoch: ",epoch+1)
        #model.train()
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss 
        start = time.time()
        # Batches
        for i, (images, gt_tl_heat, gt_br_heat, gt_mask, gt_tl_regr, gt_tl_regr) in enumerate(trainloader):
            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            gt_tl_heat = gt_tl_heat.to(device)
            gt_br_heat = gt_br_heat.to(device)
            gt_mask = gt_mask.to(device)
            gt_tl_regr = gt_tl_regr.to(device)
            gt_tl_regr = gt_tl_regr.to(device)
            
            #boxes = boxes.to(device)
            #labels = labels.to(device)
            #print(images, boxes, labels)
            #boxes = [b.to(device) for b in boxes]
            #labels = [l.to(device) for l in labels]

            # Forward prop.
            #predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            outs = model(images)
            print(outs)
            # Loss
            loss = criterion(outs, outs)  # scalar

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Update model
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)
            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(trainloader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))
        del outs, images, gt_tl_heat, gt_br_heat, gt_mask, gt_tl_regr, gt_tl_regr  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
