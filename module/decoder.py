import torch.nn as nn
import torch


class Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        self.deconv_with_bias = False

        # ----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
        # ----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(num_layers=3, num_filters=[256, 128, 64], num_kernels=[4, 4, 4])

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            layers.append(
                nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=planes, kernel_size=kernel, stride=2,
                                   padding=1, output_padding=0, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(Head, self).__init__()
        # -----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        # -----------------------------------------------------------------#

        # 热力图预测部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0))

        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

        # 中心点预测的部分
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return hm, wh, offset
