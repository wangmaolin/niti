import torch
import torch.nn as nn

cfg = {
    9:  [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

BIAS_FLAG = False

class VGG_cifar(nn.Module):
    def __init__(self, depth,num_classes):
        super(VGG_cifar, self).__init__()
        self.layers = self._make_layers(cfg[depth])
        self.classifier = nn.Sequential(
            nn.Linear(512,num_classes,bias=BIAS_FLAG),
        )
        self._initialize_weights()
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 0.01,
             'momentum': 0.9,'weight_decay': 5e-4 },
            {'epoch': 100, 'lr': 0.001}]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                    # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1,bias=BIAS_FLAG),
                           # nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

class VGG_imagenet(nn.Module):
    def __init__(self, depth):
        super(VGG_imagenet, self).__init__()
        self.layers= self._make_layers(cfg[depth])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
        )
        self._initialize_weights()
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-2,
             'momentum': 0.9,'weight_decay': 5e-4 },
            # {'epoch': 0, 'optimizer': 'Adam', 'lr': 1e-3},
            {'epoch': 30, 'lr': 1e-3},
            {'epoch': 60, 'lr': 1e-4}]

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)