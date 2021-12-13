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
        self.bb = []
        self.bb.append([self.min_x, self.min_y, self.max_x, self.max_y])
        del img

    def __getitem__(self, idx):

        img = self.ct[idx]
        label = self.label[idx]
        min_x = self.min_x[idx]
        min_y = self.min_y[idx]
        max_x = self.max_x[idx]
        max_y = self.max_y[idx]
        boxes = []
        boxes.append([min_x, min_y, max_x, max_y])

        #boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #label = torch.ones((label,), dtype=torch.int64)
        #area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = ((max_x - min_x) * (max_y - min_y))
        
        target = {}
        target["boxes"] = boxes
        target["label"] = label
        target["area"] = area

        """
        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)
            #img, target = self.transforms(img, target)
        """
        if self.transforms is not None:
            img = self.transforms(img)
            boxes = self.transforms(boxes)
            label = self.transforms(label)
            #area = self.transforms(area)
            
        return img, boxes, label

    def __len__(self):
        return len(self.ct)
    
    def collate_fn(self, batch):
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            

        images = torch.stack(images, dim=0)

        return images, boxes, labels # tensor (N, 3, 300, 300), 3 lists of N tensors each



"""
class my_dataset(Dataset):
    def __init__(self, load_dir, transforms=None):
        super().__init__()
        self.transform = transforms
        img = np.load(load_dir)
        self.ct = img['ct']
        #self.label = img['label']
        self.min_x = img['min_x']
        self.min_y = img['min_y']
        self.max_x = img['max_x']
        self.max_y = img['max_y']
        #np.array([[min_x,min_y],[max_x, max_y]])
        self.bb = []
        self.bb.append([self.min_x, self.min_y, self.max_x, self.max_y])

        del img

    def __getitem__(self, index):
        img = self.ct[index]
        target = self.bb[index]
        #bbox = torch.as_tensor(self.bb, dtype=torch.float32)
        #target = {}
        #target["label"] = self.label[index]
        #target["bbox"] = self.bb[index]
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
        return img, target

    def __len__(self):
        return len(self.ct)
"""

######################################################################
###########################  original ################################
######################################################################

"""
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages")))) ###list로 들어감
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks")))) ###list로 들어감
 
    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
 
        mask = np.array(mask) #인덱스 마스크 하나 오픈
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
 
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
 
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
    def __len__(self):
        return len(self.imgs)
"""


###################################################
################## tmp labeling ###################
###################################################

"""
class my_dataset(Dataset):
    def __init__(self, load_dir, transforms=None):
        super().__init__()
        self.transform = transforms
        img = np.load(load_dir)
        self.ct = img['ct']
        self.bb = img['bb']
        del img

    def __getitem__(self, index):
        img = self.ct[index]
        target = self.bb[index]
        #bbox = torch.as_tensor(self.bb, dtype=torch.float32)
        #target = {}
        #target["label"] = self.label[index]
        #target["bbox"] = self.bb[index]
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
        return img, target

    def __len__(self):
        return len(self.ct)
"""