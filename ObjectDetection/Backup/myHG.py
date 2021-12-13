#from typing_extensions import Unpack
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, block=Residual, num_stacks=2, num_blocks=4, num_classes=1):
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

"""
    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out
"""



# channels, height, width

class Top_Left_Corner(nn.Module):
    def __init__(self, in_channels, img_size):
        super(Top_Left_Corner, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        
    def _top_pool(self,x):
        for i in range(x.size(dim=0)):
            for k in range(x.size(dim=2)):
                for j in range(x.size(dim=1)-1, 0, -1):
                    if x[i][j][k] > x[i][j-1][k]:
                        x[i][j-1][k] = x[i][j][k]
        return x

    def _left_pool(self,x):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                for k in range(x.size(dim=2)-1, 0, -1):
                    if x[i][j][k] > x[i][j][k-1]:
                        x[i][j][k-1] = x[i][j][k]
        return x

    def forward(self, x):
        top = self._top_pool(x)
        left = self._left_pool(x)
        out = top + left
        return out

class Bottom_Right_Corner(nn.Module):
    def __init__(self, in_channels, img_size):
        super(Bottom_Right_Corner, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size

    def _bottom_pool(self,x):
        for i in range(x.size(dim=0)):
            for k in range(x.size(dim=2)):
                for j in range(x.size(dim=1)-1):
                    if x[i][j][k] > x[i][j+1][k]:
                        x[i][j+1][k] = x[i][j][k]
        return x

    def _right_pool(self,x):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                for k in range(x.size(dim=2)-1):
                    if x[i][j][k] > x[i][j][k+1]:
                        x[i][j][k+1] = x[i][j][k]
        return x        
        
    def forward(self, x):
        bottom = self._bottom_pool(x)
        right = self._right_pool(x)
        out = bottom + right
        return out


"""
class Top_Pool(nn.Module):
    def __init__(self, in_channels, img_size):
        super(Top_Pool, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size

    def forward(self, x):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                x = x
        return x

class Left_Pool(nn.Module):
    def __init__(self):
        super(Left_Pool, self).__init__()
        
    def forward(self, x):
        for i in range(x.size(dim=0)):
            for j in range(x.size(dim=1)):
                for k in range(x.size(dim=2)-1, 0, -1):
                    if x[i][j][k] > x[i][j][k-1]:
                        x[i][j][k-1] = x[i][j][k]
        return x

   
class Bottom_Pool(nn.Module):
    def __init__(self, in_channels):
        super(Bottom_Pool, self).__init__()
        self.channels = in_channels
    
    def forward(self, x):
        return x

class Right_Pool(nn.Module):
    def __init__(self, in_channels):
        super(Right_Pool, self).__init__()
        self.channels = in_channels
    
    def forward(self, x):
        return x


class Bottom_Right_Corner(nn.Module):
    def __init__(self, in_channels, img_size):
        super(Bottom_Right_Corner, self).__init__()
        self.in_channels = in_channels
        self.img_size = img_size

        self.bottom_pool = Bottom_Pool()
        self.right_pool = Right_Pool()
        
    def forward(self, x):
        bottom = self.bottom_pool(x)
        right = self.right_pool(x)
        out = bottom + right
        return out
"""        