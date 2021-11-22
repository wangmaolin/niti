from torch.nn.modules import Sequential
from ti_torch import TiLinear
from ti_torch import TiReLU
from ti_torch import TiConv2d
from ti_torch import TiDropout 
from ti_torch import TiMaxpool2d
from ti_torch import TiFlat
from ti_net import TiNet

cifar_cfg = {
    7:[128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    8:[128, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    9:[128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'M'],
}

class TiVGG_cifar(TiNet):
    def __init__(self, depth = None):
        super(TiVGG_cifar, self).__init__()
        self.forward_layers = self._make_layers(cifar_cfg[depth])

        self.regime = [{'epoch': 0,'gb':5},
                       {'epoch': 100,'gb':4},
                       {'epoch': 150,'gb':3}]

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        image_size = 32 
        for idx,x in enumerate(cfg):
            if x == 'M':
                layers += [TiMaxpool2d(2, 2)]
                image_size = image_size // 2
            else:
                layers += [TiConv2d(in_channels, x, first_layer=(idx==0)),
                           TiReLU()]
                in_channels = x

        layers += [TiFlat(),
                   TiDropout(),
                   TiLinear(512*image_size*image_size, 10, last_layer=True)]

        return Sequential(*layers)