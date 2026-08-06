import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, expansion=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(p=0.3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
                nn.Dropout(p=0.3)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out += residual
        out = self.relu(out)
        return out

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(BottleneckBlock, self).__init__()
        self.expansion = expansion
        base_channels = out_channels // expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.dropout1 = nn.Dropout(p=0.3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(base_channels)
        self.dropout2 = nn.Dropout(p=0.3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(base_channels)
        self.dropout3 = nn.Dropout(p=0.3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(p=0.3)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.bn1(x)
        out = self.dropout1(out)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.dropout3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out += residual
        return out

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        ### YOUR CODE HERE
        self.args = args

        self.stage_filters = [64, 128, 256]
        if args.resnet_version == 1:
            self.expansion = BasicBlock.expansion
            self.block = BasicBlock
        elif args.resnet_version == 2:
            self.expansion = BottleneckBlock.expansion
            self.block = BottleneckBlock
        else:
            raise ValueError("Invalid ResNet version")

        # Initial layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout_init = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(inplace=True)

        # Stages
        self.stage1 = self._make_stage(16, self.stage_filters[0], args.resnet_size, stride=1)
        self.stage2 = self._make_stage(self.stage_filters[0], self.stage_filters[1], args.resnet_size, stride=2)
        self.stage3 = self._make_stage(self.stage_filters[1], self.stage_filters[2], args.resnet_size, stride=2)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout_final = nn.Dropout(p=args.drop)
        self.fc = nn.Linear(self.stage_filters[-1], args.num_classes)

    def _make_stage(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(self.block(in_channels, out_channels, stride, self.expansion))
        for _ in range(1, n_blocks):
            layers.append(self.block(out_channels, out_channels, 1, self.expansion))
        return nn.Sequential(*layers)
        ### END YOUR CODE

    def forward(self, x):
        '''
        Input x: a batch of images (batch size x 3 x 32 x 32)
        Return the predictions of each image (batch size x 10)
        '''
        ### YOUR CODE HERE
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout_init(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_final(x)
        x = self.fc(x)
        ### END YOUR CODE
        return x
