
import torch.nn as nn

norm_mean, norm_var = 1.0, 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,cp_rate=[0.], tmp_name=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1.cp_rate = cp_rate[0]
        self.conv1.tmp_name = tmp_name
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.conv2.cp_rate = cp_rate[1]
        self.conv2.tmp_name = tmp_name
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,  covcfg=None,compress_rate=None):
        self.inplanes = 64
        self.covcfg = covcfg
        super(ResNet, self).__init__()

        self.compress_rate = compress_rate
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv1.cp_rate = compress_rate[0]
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, cp_rate=compress_rate[1:5], tmp_name='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cp_rate=compress_rate[5:10], tmp_name='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cp_rate=compress_rate[10:15], tmp_name='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cp_rate=compress_rate[15:20], tmp_name='layer4')
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, cp_rate, tmp_name):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        if downsample != None :
            u = 1
        else : u = 0
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,cp_rate=cp_rate[u:u+2], tmp_name=tmp_name + '_block' + str(1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,cp_rate=cp_rate[u+2:u+4], tmp_name=tmp_name + '_block' + str(i + 1)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

def resnet_18(dataset,compress_rate=None):
    cov_cfg = [(3*i + 3) for i in range(2*2 + 1 + 2*2 + 1 + 2*2 + 1 + 2*2 + 1 + 1)]
    model = ResNet(BasicBlock, [2, 2, 2, 2], covcfg=cov_cfg, compress_rate=compress_rate)
    return model