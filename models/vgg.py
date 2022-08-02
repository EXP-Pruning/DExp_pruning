
import math
import torch.nn as nn
from collections import OrderedDict


norm_mean, norm_var = 0.0, 1.0
#conv3-64×2-->maxpool-->conv3-128×2-->maxpool-->conv3-256×3-->maxpool-->conv3-512×3-->maxpool-->conv3-512×3-->maxpool
defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]   #relu层在总层数的位置
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]   #卷积层在总层数的位置


class VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, cfg=None, dataset=None,compress_rate=None):
        super(VGG, self).__init__()
        self.features = nn.Sequential()
        if dataset == 'cifar10' :
            num_classes = 10
        else: num_classes = 100
        if cfg is None:
            cfg = defaultcfg

        self.relucfg = relucfg
        self.covcfg = convcfg
        self.compress_rate = compress_rate
        self.features = self.make_layers(cfg[:-1], True, compress_rate)    #所有层 并且附加压缩率  True表示加入BN
        # 下面是全连接层用于分类(512)-->(512)-->10
        self.classifier = nn.Sequential(OrderedDict([         #有序字典  记录插入顺序
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=True, compress_rate=None):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # 输入通道(输入通道, 输出通道, kernel_size=3, padding=1)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                conv2d.cp_rate = compress_rate[cnt]
                cnt += 1

                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                # 加入relu层，inplace=True表示的是上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v  #输出通道变输入
        # 卷积层定义完毕，返回
        return layers

    def forward(self, x):
        # 组装,卷积计算输入[1,3,32,32]-->[1, 512, 2, 2]
        x = self.features(x)
        # 平均值池化[1, 512, 2, 2]-->[1, 512, 2, 2]
        x = nn.AvgPool2d(2)(x)
        # 形状调整[1, 512, 2, 2]-->[1,512]
        x = x.view(x.size(0), -1)
        # 送入卷积层
        x = self.classifier(x)
        # 输出
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                   #判断是否为Conv2d
                # 卷积核尺度与输出通道相乘
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels    # 数量* 维数* 输出通道数
                # 权重的初始化
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # 偏差的初始化，填充0
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg_16_bn(dataset,compress_rate=None):
    return VGG(dataset=dataset,compress_rate=compress_rate) #添加的 卷积指定索引位置和对应卷积层的压缩率， 模型结构没有任何变化，只是增加了类变量，便于 第二步main函数 调用 每层压缩信息，

