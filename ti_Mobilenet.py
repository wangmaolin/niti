import torch
import torch.nn as nn
from ti_torch import TiLinear
from ti_torch import TiReLU
from ti_torch import TiConv2d
from ti_torch import TiDropout 
from ti_torch import TiMaxpool2d
# from ti_torch import TiAdaptiveAvgpool2d
from ti_torch import TiFlat
from ti_net import TiNet

BITWIDTH = 7

def TiFloatToInt8(input):
    '''
    Convert float tensors to integer tensors, used to
    1. feed in data in forward pass
    2. feed in loss in backwards
    '''
    input_range = torch.max(torch.abs(input))
    input_bitwidth = torch.ceil(torch.log2(input_range))
    act_exp = input_bitwidth - BITWIDTH
    """ old quantization function, may introduce a small scale to the quantized value"""
    # round_val = torch.round(input/input_range*(2**BITWIDTH-1)).type(torch.int8).to(GPU)
    norm_val = input*(2**(BITWIDTH-input_bitwidth))
    round_val = torch.round(norm_val)
    clamp_val = torch.clamp(round_val,-128, 127).type(torch.int8).to('cuda')
    # return quantised int8 and exponent
    return clamp_val, act_exp

def TiInt8ToFloat(input):
    '''
    Convert int tensors to float tensors
    Could be used to verify the functionality of integer NN
    '''
    act_in, exp_in = input
    return act_in.float() * (2**(exp_in.float()))

cifar_cfg = {
    7:[128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    8:[128, 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    9:[128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'M'],
}

class Depthwise_Conv(nn.Module):
    def __init__(self, in_channels, stride=(1,1)):
        super(Depthwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                in_channels,
                kernel_size=(3,3),
                stride=stride, 
                padding=(1, 1), 
                groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

class Pointwise_Conv(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super(Pointwise_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, input_image):
        x = self.conv(input_image)
        return x

class Depthwise_Separable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(Depthwise_Separable_Conv, self).__init__()
        self.dw = Depthwise_Conv(in_channels=in_channels, stride=stride)
        self.pw = Pointwise_Conv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, input_image): 
        x = self.pw(self.dw(input_image))
        return x

class DepthSeparableConv2d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # self.depthwise = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels,
        #         in_channels,
        #         kernel_size,
        #         groups = in_channels,
        #         **kwargs
        #     ),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(inplace=True)
        # )

        # self.pointwise = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

        self.convDW = TiConv2d(
            in_channels,
            in_channels,
            3
        )

        self.convPW = TiConv2d(
            in_channels,
            out_channels,
            1
        )
    
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = TiReLU()

    def forward(self, x):
        x_val, x_exp = self.convDW(x)
        x = (x_val, x_exp)
        x = TiInt8ToFloat(x)
        x = self.bn1(x)
        x_val, x_exp = TiFloatToInt8(x)
        x = (x_val, x_exp)
        x_val, x_exp = self.relu(x)
        x = (x_val, x_exp)

        x_val, x_exp = self.convPW(x)
        x = (x_val, x_exp)
        x = TiInt8ToFloat(x)
        x = self.bn2(x)
        x_val, x_exp = TiFloatToInt8(x)
        x = (x_val, x_exp)
        x_val, x_exp = self.relu(x)
        # x = self.depthwise(x)
        # x = self.pointwise(x)

        return x_val, x_exp

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        # self.conv = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size, **kwargs
        # )
        # self.relu = nn.ReLU(inplace=True)
        self.conv = TiConv2d(
            in_channels,
            out_channels,
            kernel_size
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = TiReLU()

    def forward(self, x):
        print(x.size())
        x_val, x_exp = self.conv(x)
        x = [x_val, x_exp]
        x = TiInt8ToFloat(x)
        x = self.bn(x)
        x_val, x_exp = TiFloatToInt8(x)
        x = [x_val, x_exp]
        x_val, x_exp = self.relu(x)

        return x_val, x_exp

class TiMobileNetV1_cifar(TiNet):
    def __init__(self, width_multiplier=1, class_num=10) -> None:
        super(TiMobileNetV1_cifar, self).__init__()
        # self.forward_layers = self._make_layers()

        self.regime = [{'epoch': 0, 'gb':5},
                       {'epoch': 100, 'gb':4},
                       {'epoch': 150, 'gb':3}]

        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
            DepthSeparableConv2d(
                int(32 * alpha),
                int(64 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        self.conv1 = nn.Sequential(
            DepthSeparableConv2d(
                int(64 * alpha),
                int(128 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(128 * alpha),
                int(128 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        self.conv2 = nn.Sequential(
            DepthSeparableConv2d(
                int(128 * alpha),
                int(256 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(256 * alpha),
                int(256 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        self.conv3 = nn.Sequential(
            DepthSeparableConv2d(
                int(256 * alpha),
                int(512 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        self.conv4 = nn.Sequential(
            DepthSeparableConv2d(
                int(512 * alpha),
                int(1024 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeparableConv2d(
                int(1024 * alpha),
                int(1024 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            )
        )

        # self.fc = nn.Linear(int(1024 * alpha), class_num)
        self.fc = TiLinear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.avg = TiAdaptiveAvgpool2d()

    def forward(self, x): 
        x_val, x_exp = self.stem(x)
        x = (x_val, x_exp)

        x_val, x_exp = self.conv1(x)
        x = (x_val, x_exp)
        x_val, x_exp = self.conv2(x)
        x = (x_val, x_exp)
        x_val, x_exp = self.conv3(x)
        x = (x_val, x_exp)
        x_val, x_exp = self.conv4(x)
        x = (x_val, x_exp)

        x = TiInt8ToFloat(x)
        x = self.avg(x)
        x_val, x_exp = TiFloatToInt8(x)
        x = (x_val, x_exp)
        # x = x.view(x.size(0), -1)
        x_val, x_exp = self.fc(x)

        return x_val, x_exp

    # def _make_layers(self):
    #     layers = []
    #     in_channels = 3
    #     image_size = 32
    #     layers += [
    #         self.stem(),

    #         self.conv1(),
    #         self.conv2(),
    #         self.conv3(),
    #         self.conv4(),

    #         self.avg(),
    #         self.fc()
    #     ]

    #     return nn.Sequential(*layers)

def mobilenet(alpha=1, class_num=10):
    return TiMobileNetV1_cifar(alpha, class_num)


# class MobileNet(nn.Module):

#     def __init__(self, in_channels=3, num_filter=32, num_classes=1000):
#         super(MobileNet, self).__init__()

#         self.conv == nn.Sequential(
#             nn.Conv2d(
#                 in_channels, 
#                 kernel_size=(3,3),
#                 stride=(2,2),
#                 padding=(1,1)  
#             ),
#             nn.BatchNorm2d(num_filter),
#             nn.ReLU(inplace=True)
#         )

#         self.in_channels = num_filter

#         self.nlayer_filter = [
#             num_filter * 2,

#         ]

#         self.DSC = self.layer_construct()

#         self.avgpool = nn.AdaptiveAvgPool2d()
#         self.fc == nn.Sequential(
#             nn.Linear(1024, num_classes),
#             nn.Softmax()
#         )

#     def forward(self, input_image): 
#         N = input_image.shape[0]
#         x = self.conv(input_image)
#         x = self.DSC(x)
#         x = self.avgpool(x)
#         x = x.reshape(N, -1)
#         x = self.fc(x)
#         return x
