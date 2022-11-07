import math
from torch import nn

from module.decoder import Decoder, Head
from module.resnet import resnet18, resnet34, resnet50


class CenterNetResnet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, pretrained: bool):
        super(CenterNetResnet, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        self.pretrained = pretrained

        self.backbone = None
        self.decoder = None
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
            self.decoder = Decoder(512)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=pretrained)
            self.decoder = Decoder(512)
        else:
            self.backbone = resnet50(pretrained=pretrained)
            self.decoder = Decoder(2048)
        # -----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        # -----------------------------------------------------------------#
        self.head = Head(channel=64, num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.head.cls_head[-1].weight.data.fill_(0)
        self.head.cls_head[-1].bias.data.fill_(-2.19)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))


def center_net_resnet18(num_classes: int, pretrained: bool):
    model = CenterNetResnet('resnet18', num_classes, pretrained)
    return model


def center_net_resnet34(num_classes: int, pretrained: bool):
    model = CenterNetResnet('resnet34', num_classes, pretrained)
    return model


def center_net_resnet50(num_classes: int, pretrained: bool):
    model = CenterNetResnet('resnet50', num_classes, pretrained)
    return model
