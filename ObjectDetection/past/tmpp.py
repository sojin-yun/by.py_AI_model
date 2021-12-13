from typing_extensions import Unpack
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        #residual function
        self.residual_function = nn.Sequential(
            nn.Relu(), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.Relu(), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.Relu(), nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )
        #shortcut
        self.skip_layer = nn.Sequential(
            nn.ReLU(), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )
        #if in_channels == out_channels:
        #    self.need_skip = False
        #else:
        #    self.need_skip = True
    
    def forward(self, x):
        #if self.need_skip:
        #    residual = self.skip_layer(x)
        #else:
        #    residual = x
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
        #self.conv_layer = Multiscale_residual_block(input_ch * 2, input_ch * 2)
        #self.conv_final = nn.Conv2d(input_ch * 2, num_landmarks, 1)

    def _conv_layer(self, in_feats, out_feats):
        conv = nn.Conv2d(in_feats, out_feats, kernel_size=1, bias=False)
        return conv


    def forward(self, x):
        downsample = []
        for i in range(self.num_blocks):            
            hg_block = Residual(x, self.channels[i], self.channels[i+1], stride=2)
            downsample.append(hg_block)
            x = hg_block
        hg_block = Residual(x, self.channels[-2], self.channels[-1], stride=2)
        x = hg_block
        # upsample
        for i in range(self.num_blocks+1, 1, -1):
            upsample = self.upsample_layer(x)
            upsample = self._conv_layer(upsample, self.channels[i], self.channels[i-1])
            out = downsample[i] + upsample
            x = out
        return out


class Classifier(nn.Module):
  def __init__(self, num_landmarks, featdim):
    super(Classifier, self).__init__()
    self.conv1 = nn.Conv2d(num_landmarks, featdim, 5) 
    self.conv2 = nn.Conv2d(featdim, featdim * 2, 5)

    self.fc1 = nn.Linear(featdim * 2 * 29 * 29, 512) # 29=(((((128-5)+1)/2)-5)+1)/2
    self.fc2 = nn.Linear(512, 2)

  def forward(self, x):
    # x = F.max_pool2d(F.relu(self.conv1(x)), 2) 
    # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = F.max_pool2d(self.conv1(x), 2) 
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 128 * 29 * 29)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    return F.log_softmax(x, dim = 1)


class HourglassNet(nn.Module):
    def __init__(self, block=Residual, num_stacks=2, num_blocks=4, num_classes=1):
        super(HourglassNet, self).__init__()
        self.in_channels = 64
        self.num_feats = 256
        self.num_stacks = num_stacks
        
        # Initial processing of the image (gpu 사용량이 높아서)
        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = block(self.in_channels, int(self.num_feats/2))
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.layer2 = block(int(self.num_feats/2), int(self.num_feats/2))
        self.layer3 = block(int(self.num_feats/2), self.num_feats)
        self.hg_block = HourglassModule(self.num_feats, num_blocks=num_blocks, num_classes=num_classes) 

        #self.classifier = Classifier(num_landmarks=num_landmarks, featdim=in_channels)
################classifier의 landmark가 뭘까??



##########score를 내보내야한다#############
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        hourglass_return = self.hg_block(x) 

        if self.is_classfy:
            classifier_return = self.classifier(hourglass_return)
            return hourglass_return, classifier_return
        else:
            return hourglass_return

################참고
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



class ckarh_HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []

        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)


    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv, bn, self.relu,
            )

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
