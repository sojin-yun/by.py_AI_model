import numpy as np
from torch.utils.data import Dataset

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