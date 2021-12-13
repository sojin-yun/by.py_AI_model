import torch
import torch.nn as nn
from torchsummary import summary

class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        out_channels = growth_rate
        self.BRC_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
    def forward(self, x):
        return torch.cat([x, self.BRC_layer(x)], 1)

class RenseBlock(nn.Module):
    def __init__(self, block, in_channels, nblocks, growth_rate):
        super().__init__()
        
        inner_channel = in_channels
        out_channels = in_channels + nblocks * growth_rate

        self.rense_block = nn.Sequential()
        for index in range(nblocks):
            self.rense_block.add_module('basic_block_layer_{}'.format(index), block(inner_channel, growth_rate))
            inner_channel += growth_rate

        self.skip_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, x):
        return self.rense_block(x) + self.skip_connection(x)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )
    def forward(self, x):
        return self.down_sample(x)

class RenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, compression=0.5, num_classes=2):
        """
        block: BasicBlock, Bottleneck
        num_block: the number of basicblks(or bottleN) in each renseblk (renseblk은 우선 3으로, 4로 할 지는 추후에 생각)
        """
        super().__init__()

        self.growth_rate = growth_rate
        self.compression = compression
        inner_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(1, inner_channels, kernel_size=7, padding=3, stride=2, bias=False)
        self.avg_pool = nn.AvgPool2d(3, stride=2)
        

        """
        # 1. conv stride = 2^len(nblocks)-1
        glob_inner_channels = inner_channels
        for index in range(len(nblocks)-1):
            glob_inner_channels = int((glob_inner_channels + growth_rate * nblocks[index]) / 2)
        glob_inner_channels += growth_rate * nblocks[-1]
        self.glob_features = nn.Conv2d(inner_channels, glob_inner_channels, kernel_size=1, stride=2**(len(nblocks)-1), bias=False)
        """
        """
        # 2. len(nblocks)-1번 AVGPOOL
        #glob_in_channels = int(inner_channels / 2)
        #glob_inner_channels = glob_in_channels
        glob_inner_channels = inner_channels
        self.glob_features = nn.Sequential()
        for index in range(len(nblocks)-1):
            glob_inner_channels = int((glob_inner_channels + growth_rate * nblocks[index]) / 2)
            self.glob_features.add_module("avg_pool_{}".format(index), nn.AvgPool2d(2, stride=2))
        glob_inner_channels += growth_rate * nblocks[-1]
        self.glob_features.add_module("glob_conv_layer", nn.Conv2d(inner_channels, glob_inner_channels, kernel_size=1, bias=False))
        """

        # 3. len(nblocks)-1번 MAXPOOL
        #glob_in_channels = int(inner_channels / 2)
        #glob_inner_channels = glob_in_channels
        glob_inner_channels = inner_channels
        self.glob_features = nn.Sequential()
        for index in range(len(nblocks)-1):
            glob_inner_channels = int((glob_inner_channels + growth_rate * nblocks[index]) / 2)
            self.glob_features.add_module("avg_pool_{}".format(index), nn.MaxPool2d(2, stride=2))
        glob_inner_channels += growth_rate * nblocks[-1]
        self.glob_features.add_module("glob_conv_layer", nn.Conv2d(inner_channels, glob_inner_channels, kernel_size=1, bias=False))
        

        self.features = nn.Sequential()
        for index in range(len(nblocks)-1):
            self.features.add_module("rense_block_layer_{}".format(index), RenseBlock(block, inner_channels, nblocks[index], growth_rate))
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(compression * inner_channels)
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels
        self.features.add_module("rense_block_layer_{}".format(len(nblocks)), RenseBlock(block, inner_channels, nblocks[-1], growth_rate))
        inner_channels += growth_rate * nblocks[-1]



        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.linear = nn.Linear(inner_channels, num_classes)
        self.linear = nn.Sequential( 
            nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(int(inner_channels/2), num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        output = self.conv1(x)
        output = self.avg_pool(output)
        #output1 = self.avg.pool(output)
        #output2 = self.avg.pool(output1)
        output = self.features(output) + self.glob_features(output)
        output = self.glob_pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RenseNet(BasicBlock, [3,3,3], growth_rate=12).to(device)
summary(net, (1, 512, 512))