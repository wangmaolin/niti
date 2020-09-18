import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.modules import Sequential
from torch.nn import functional as F
from ti_torch import TiLinear
from ti_torch import TiReLU
from ti_torch import TiConv2d, TiConv2d_acc23
from ti_torch import TiMaxpool2d
from ti_torch import TiFlat
from ti_torch import TiInt8ToFloat
from ti_torch import TiFloatToInt8
from ti_torch import PstoShiftInt32
from ti_torch import Int8Tensor
from ti_torch import int8_clip
from ti_torch import Int32Tensor
from vgg import cfg 
from ti_net import TiNet

BIAS_FLAG = False

class TiVGG_cifar(TiNet):
    def __init__(self, depth,num_classes):
        super(TiVGG_cifar, self).__init__()
        self.data2int=TiFloatToInt8()
        self.forward_layers = self._make_layers(cfg[depth],num_classes)
        self.regime = [
            {'epoch': 0,'gb':2},
            {'epoch': 100, 'gb': -1}]
        self.regime_frac = [
            {'epoch': 0,'gb':4},
            {'epoch': 150, 'gb': 2}]
 
    def _make_layers(self, cfg, num_classes):

        layers = []
        in_channels = 3
        for idx,x in enumerate(cfg):
            if x == 'M':
                layers += [TiMaxpool2d(2, 2)]
            else:
                layers += [
                    TiConv2d_acc23(
                        in_channels=in_channels,
                        out_channels=x,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=BIAS_FLAG,
                        first_layer=(idx==0)),
                    TiReLU()]
                in_channels = x

        layers += [TiFlat()]

        layers += [TiLinear(512,num_classes,bias=BIAS_FLAG)]
        return Sequential(*layers)