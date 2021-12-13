import os
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import wandb

device = torch.device('cuda:4')

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    run()

class DukeDataset(Dataset):

    def __init__(self,data_dir, mode=None, transform=None):
        self.transform =transform
        self.mode = mode
     
        if self.mode =='train':
            self.image_path = os.path.join(data_dir,"Train_img")
            self.label_path = os.path.join(data_dir,"Train_label")

     
        elif self.mode == 'test':
            self.image_path = os.path.join(data_dir,"Validation_img")
            self.label_path = os.path.join(data_dir,"Validation_label")


        lst_data_image = os.listdir(self.image_path)
        lst_data_label = os.listdir(self.label_path)
     
        self.lst_input = lst_data_image
        self.lst_label = lst_data_label

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self,index):

        image = np.load(os.path.join(self.image_path,self.lst_input[index]))
        label = np.load(os.path.join(self.label_path,self.lst_label[index]))


        ## 둘 다 16bit 이지만 mr영상은 max 값 편차가 크고 대부분 65535보다 훨씬 작은 값
        image = image/image.max()
        label = label/65535.0


        if image.ndim == 2:
            image = image[:,:,np.newaxis]
        if label.ndim == 2:
            label = label[:,:,np.newaxis]

        # data = {'input':image, 'label': label}
        
        if self.transform is not None:
            data = self.transform(image = image, mask = label)
            
            # image의 값을 0-1 사이로 
            data_img = (data["image"]*0.5)+0.5
            data_lab = data["mask"]
            data_lab = data_lab.permute(2,0,1)

            #data = {'input':data_img,'label':data_lab}

        return data_img, data_lab



import albumentations.augmentations as AA
import albumentations.pytorch as Ap
from albumentations.core.composition import Compose, OneOf

def transform_js(mode):
    if mode == 'train':
        train_transform = Compose([
             AA.Resize(height = 384, width = 384),
             OneOf([AA.HorizontalFlip(),
             AA.RandomRotate90(),
             AA.VerticalFlip()]), # oneof에 p 부여 가능,
             AA.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5),max_pixel_value=1.0),
             Ap.transforms.ToTensorV2()
             ])        
        return train_transform
    
    elif mode == 'test':
        test_transform = Compose([
             AA.Resize(height = 384, width = 384),
             AA.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5),max_pixel_value=1.0),
             Ap.transforms.ToTensorV2()
             ])
        return test_transform


class JUNet(nn.Module):
    def __init__(self):
        super(JUNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.init()

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        # print(pool3.size())
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                print('Conv2d')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                print('BN')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
                print('FC')


def dice_score_duke(pred, target):
    
    cla = []
    tar =[]
    non_zero = []
    zero = []
    classifyer = lambda x : 1.0 * (x>0.5)
    classified = classifyer(pred)

    non_zero_score = torch.zeros([1])
    zero_score = torch.zeros([1])

    eps = sys.float_info.epsilon

    # 인풋이 배치 단위로 들어옴 (배치크기,채널,h,w)

    batchsize, channel , height , width = classified.shape

    for i in range(batchsize):

        cla = classified[i,:,:,:]
        tar = target[i,:,:,:]


        if torch.count_nonzero(tar) > 0:
            intersec = torch.sum(cla * tar)
            total = torch.sum(cla) + torch.sum(tar)
            non_zero_score = ((intersec * 2) / total) + eps 
            non_zero += [non_zero_score]

        else:
            wrong_pixel = torch.sum(cla)
            total_zero = height * width
            zero_score = 1 - (((wrong_pixel*2) + eps) / total_zero)
            zero += [zero_score]


    if len(zero)>0:
        if len(non_zero)>0:
            non_zero_score_mean = sum(non_zero) / len(non_zero)
            zero_score_mean = sum(zero) / len(zero)

            return non_zero_score_mean.item(), zero_score_mean.item()

        else:
            non_zero_score_mean = non_zero_score
            zero_score_mean = sum(zero) / len(zero)

            return non_zero_score_mean.item(), zero_score_mean.item()


    else:
        non_zero_score_mean = sum(non_zero) / len(non_zero)
        zero_score_mean = zero_score

        return non_zero_score_mean.item(), zero_score_mean.item()





config = {
    "learning_rate" : 1e-3,
    "epoch" : 20,
    "pos_weight" : 20,
    "batch_size" : 5,
    "channel_of_img" : 3,
    "architecture" : "U-net",
    "dataset" : "duke_breast_cancer_mri"}


Naspath = "/home/NAS_mount/jsbarg"

trainset = DukeDataset(Naspath,'train',transform = transform_js('train'))

trainloader = DataLoader(trainset, batch_size = config["batch_size"],
                                          shuffle=True, num_workers=0)

testset = DukeDataset(Naspath,'test',transform = transform_js('test'))

testloader = DataLoader(testset, batch_size = config["batch_size"],
                                          shuffle=False, num_workers=0)


net = JUNet().to(device)

import torch.optim as optim

torch_weights = torch.ones([1]).to(device)
torch_weights = torch_weights*config["pos_weight"]

criterion = nn.BCEWithLogitsLoss(pos_weight = torch_weights)
optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"],weight_decay=1e-5)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

wandb.init(project='duke,unet',entity='barg',config = config, name = "pos_weight=20")

wandb.watch(net,criterion=criterion,log='all',log_freq=10,log_graph=(True))

for e in range(config["epoch"]):

  running_loss = 0.0
  validation_running_loss = 0.0
  dice_zero = []
  dice_non_zero = []
  dice_zero_val = []
  dice_non_zero_val = []

  net.train()

  for batch, (inputs, labels) in enumerate(trainloader):
   
     inputs = inputs.to(device)
     labels = labels.to(device)
     # 변화도(Gradient) 매개변수를 0으로 만들고
     optimizer.zero_grad()
     # 순전파 + 역전파 + 최적화를 한 후
     outputs = net(inputs)
     loss = criterion(outputs, labels)
     loss.backward()
     optimizer.step()

     running_loss += loss.item()
    
     non_zero , zero = dice_score_duke(outputs, labels)

     dice_non_zero += [non_zero]
     dice_zero += [zero]
    
     if (batch+1) % 100 == 0:
         if np.count_nonzero(dice_zero) > 0 :
             if np.count_nonzero(dice_non_zero) > 0 :
                 print(f"loss [{(batch+1)*len(inputs)}/{len(trainloader.dataset)}] : {loss.item()}")
                 print(f"zero_dice_score : {sum(dice_zero)/np.count_nonzero(dice_zero)}")
                 print(f"non_zero_dice_score : {sum(dice_non_zero)/np.count_nonzero(dice_non_zero)}")
             
             else:
                 print(f"loss [{(batch+1)*len(inputs)}/{len(trainloader.dataset)}] : {loss.item()}")
                 print(f"zero_dice_score : {sum(dice_zero)/np.count_nonzero(dice_zero)}")
                 print(f"non_zero_dice_score : 0 (non zero slice did'nt exist ")


         else:
             print(f"loss [{(batch+1)*len(inputs)}/{len(trainloader.dataset)}] : {loss.item()}")
             print(f"zero_dice_score : 0 (zero slice did'nt exist)")
             print(f"non_zero_dice_score : {sum(dice_non_zero)/np.count_nonzero(dice_non_zero)}")

  
  else:
    with torch.no_grad():
      net.eval() 
      for val_input, val_label in testloader:
                
         val_input = val_input.to(device)
         val_label = val_label.to(device)
         val_outputs = net(val_input)
         val_loss = criterion(val_outputs, val_label)

         validation_running_loss += val_loss.item()
         non_zero , zero = dice_score_duke(val_outputs, val_label)

         dice_non_zero_val += [non_zero]
         dice_zero_val += [zero]
      
  
    epoch_loss = running_loss / len(trainloader)
    val_epoch_loss = validation_running_loss / len(testloader)
    accuracy_zero = sum(dice_zero)/np.count_nonzero(dice_zero)
    accuracy_nonzero = sum(dice_non_zero)/np.count_nonzero(dice_non_zero)
    accuracy_zero_val = sum(dice_zero_val)/np.count_nonzero(dice_zero_val)
    accuracy_nonzero_val = sum(dice_non_zero_val)/np.count_nonzero(dice_non_zero_val)


    wandb.log({"train loss":epoch_loss,"validation loss":val_epoch_loss,
       "zero slice accuracy":accuracy_zero, "nonzero slice accuracy":accuracy_nonzero,
       "val_zero slice accuracy":accuracy_zero_val, "val_nonzero slice accuracy":accuracy_nonzero_val})

    print("===================================================")
    print("epoch: ", e + 1)
    print("training loss: {:.5f}".format(epoch_loss))
    print("validation loss: {:.5f}".format(val_epoch_loss))
    print("zero slice accuracy: {:.5f}".format(accuracy_zero))
    print("nonzero slice accuracy: {:.5f}".format(accuracy_nonzero))
    print("val_zero slice accuracy: {:.5f}".format(accuracy_zero_val))
    print("val_nonzero slice accuracy: {:.5f}".format(accuracy_nonzero_val))
    print("===================================================")


PATH = './junet_211015_w20.pth'
torch.save(net.state_dict(), PATH)