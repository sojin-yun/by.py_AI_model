import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class my_Dataset(Dataset):
    def __init__(self, load_dir, transforms=None):

        self.transforms = transforms
        img = np.load(load_dir)
        self.ct = img['ct']
        self.label = img['label']
        self.min_x = img['min_x']
        self.min_y = img['min_y']
        self.max_x = img['max_x']
        self.max_y = img['max_y']
        del img

    def __getitem__(self, idx):

        img = self.ct[idx]
        label = self.label[idx]
        min_x = self.min_x[idx]
        min_y = self.min_y[idx]
        max_x = self.max_x[idx]
        max_y = self.max_y[idx]
        box = []
        box.append([min_x, min_y, max_x, max_y])
        
        # x.size() = torch.size[2,256,128,128]
        
        
        gt_tl_heat = np.zeros((128,128))
        gt_tl_heat[min_x][min_y] = 1
        gt_br_heat = np.zeros((128,128))
        gt_br_heat[max_x][max_y] = 1
        gt_mask    = 
        gt_tl_regr = 
        gt_br_regr =
        

        box = torch.as_tensor(box, dtype=torch.float32)
        label = torch.ones((label,), dtype=torch.int64)

    
        #area = (box[3] - box[1]) * (box[2] - box[0])
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #area = ((max_x - min_x) * (max_y - min_y))

        if self.transforms is not None:
            img = self.transforms(img)


        return img, gt_tl_heat, gt_br_heat, gt_mask, gt_tl_regr, gt_tl_regr

    def __len__(self):
        return len(self.ct)
    
    
######################################################################
###########################  original ################################
######################################################################

"""
    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()
        #areas = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            #areas.append(b[3])
            
        #images = torch.stack(images, dim=0)
        return images, boxes, labels # tensor (N, 3, 300, 300), 3 lists of N tensors each
"""

