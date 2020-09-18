import torch
from torch.nn.modules import Module
from torch.nn.modules import Sequential
from ti_torch import TiLinear
from ti_torch import TiReLU
from ti_torch import TiMaxpool2d
from ti_torch import TiFlat
from ti_torch import TiInt8ToFloat
from ti_torch import TiFloatToInt8
from ti_torch import TiConv2d, TiConv2d_acc23
from ti_net import TiNet

class TiLenet(TiNet):
    def __init__(self):
        super(TiLenet, self).__init__()
        self.data2int=TiFloatToInt8()
        self.forward_layers = self._make_layers()
        self.regime = [
            {'epoch': 0,'gb':0}]

    def _make_layers(self):
        layers = []
        layers += [
            TiConv2d_acc23(
                in_channels=1,
                out_channels=20,
                kernel_size=5,
                stride=1,
                bias=False),
            TiReLU(),
            TiMaxpool2d(2, 2)]
        layers += [
            TiConv2d_acc23(
                in_channels=20,
                out_channels=50,
                kernel_size=5,
                stride=1,
                bias=False),
            TiReLU(),
            TiMaxpool2d(2, 2)]
        layers += [TiFlat()]
        layers += [TiLinear(
            in_features=800,
            out_features=500)]
        layers += [TiReLU()]
        layers += [TiLinear(
            in_features=500,
            out_features=10)]

        return Sequential(*layers)
