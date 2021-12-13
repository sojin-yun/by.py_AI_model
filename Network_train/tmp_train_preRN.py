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
"""
# RenseBlock(BasicBlock, in_channels, nblock[index], growth_rate)
class RenseBlock(nn.Module):
    def __init__(self, block, in_channels, nblock, growth_rate):
        super().__init__()
        self.features = nn.Sequential()
        for i in range(nblock):
            self.features.add_module('basic_block_layer_{}'.format(index), block(in_channels, growth_rate))
            in_channels += self.growth_rate

"""
class Residaul(nn.Module):
    def __init__(self, in_channels, growth_rate, nblocks):
        super().__init__()
        
        out_channels = in_channels + nblocks * growth_rate

        self.skip_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        return skip_connection

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
        self.avg_pool = nn.MaxPool2d(3, stride=2)

        self.features = nn.Sequential()

        for index in range(len(nblocks)-1):
            self.features.add_module("rense_block_layer_{}".format(index), self._make_rense_layers(block, inner_channels, nblocks[index]))
            self.features.add_module("residual_block_layer{}".format(index), self._make_skip_connection(inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(compression * inner_channels)
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))





###### 여기까지 했다......

# plt 눈금, subplot 간격

        self.rense_transit_1 = self._make_rense_transit_layers(block, inner_channels, num_block[0])
        inner_channels = int((inner_channels + num_block[0] * growth_rate) * compression)
        self.rense_transit_2 = self._make_rense_transit_layers(block, inner_channels, num_block[1])
        inner_channels = int((inner_channels + num_block[1] * growth_rate) * compression)
        self.rense_transit_3 = self._make_rense_transit_layers(block, inner_channels, num_block[2])
        inner_channels = int((inner_channels + num_block[2] * growth_rate) * compression)
        self.rense_4 = self._make_rense_layers(block, inner_channels, num_block[3])
        inner_channels = inner_channels + num_block[3] * growth_rate
        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        #self.linear = nn.Linear(inner_channels, num_classes)
        self.linear = nn.Sequential( 
            #nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(), nn.Dropout(p=0.5), 
            nn.Linear(inner_channels, int(inner_channels/2)), nn.ReLU(),
            nn.Linear(int(inner_channels/2), num_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _make_rense_layers(self, block, in_channels, nblocks):
        rense_block = nn.Sequential()
        for index in range(nblocks):
            rense_block.add_module('basic_block_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return rense_block
        
    def _make_skip_connection(self, in_channels, nblocks):
        out_channels = in_channels + nblocks * self.growth_rate
        skip_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        return skip_connection

    def _make_rense_transit_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('basic_block_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        dense_block.add_module('bn', nn.BatchNorm2d(in_channels))
        out_channels = int(self.compression * in_channels)
        dense_block.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        dense_block.add_module('pool', nn.AvgPool2d(2, stride=2))
        return dense_block

    def forward(self, x):
        output = self.conv1(x)
        output = self.avg_pool(output)
        output = self.rense_transit_1(output)
        output = self.rense_transit_2(output)
        output = self.rense_transit_3(output)
        output = self.rense_4(output)
        output = self.glob_pool(output)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output 
