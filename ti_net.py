import torch
import math
from torch.nn.modules import Module
from ti_torch import TiInt8ToFloat
from ti_torch import TiFloatToInt8
from ti_torch import TiLinear
from ti_torch import Int8Tensor
from ti_torch import Int8zeros
from ti_torch import Int32Tensor
from ti_torch import int8_clip
from ti_torch import PstoShiftInt32
from ti_torch import RoundInt32
from ti_torch import BITWIDTH
import collections
from ti_loss import TiLoss

class TiNet(Module):
    '''
    Training deep integer network
    with learning rate schedule
    '''
    def __init__(self):
        super(TiNet, self).__init__()
        self.epochs = 0
        self.loss = TiLoss()

    def forward(self,x):
        x = self.data2int(x)
        self.out, self.out_exp = self.forward_layers(x)
        return self.out, self.out_exp

    def backward(self,target):
        x = self.loss(self.out, self.out_exp, target)
        for layer in reversed(self.forward_layers):
            if isinstance(layer,torch.nn.Sequential):
                for block in reversed(layer):
                    x=block.backward(x)
            else:
                x=layer.backward(x)

    def train(self):
        pass
        # for idx,l in enumerate(self.forward_layers):
        #     if isinstance(l,TiDropout):
        #         l.training = True

    def eval(self):
        pass
        # for idx,l in enumerate(self.forward_layers):
        #     if isinstance(l,TiDropout):
        #         l.training = False

    def state_dict(self):
        state_dict=collections.OrderedDict()
        for idx,l in enumerate(self.forward_layers):
            if hasattr(l,'weight'):
                layer_prefix = 'layers.'+str(idx)+'.'
                state_dict[layer_prefix+'weight']=l.weight
                state_dict[layer_prefix+'weight_exp']=l.weight_exp
            if hasattr(l,'bias'):
                state_dict[layer_prefix+'bias']=l.bias
                state_dict[layer_prefix+'bias_exp']=l.bias_exp
        return state_dict

    def load_state_dict(self,state_dict):
        for idx,l in enumerate(self.forward_layers):
            if hasattr(l,'weight'):
                layer_prefix = 'layers.'+str(idx)+'.'
                l.weight.data=state_dict[layer_prefix+'weight']
                l.weight_exp.data=state_dict[layer_prefix+'weight_exp']
                l.weight_frac.data=l.weight.type(torch.int32)*2**l.frac_bits

            if hasattr(l,'bias'):
                layer_prefix = 'layers.'+str(idx)+'.'
                l.bias.data=state_dict[layer_prefix+'bias']
                l.bias_exp.data=state_dict[layer_prefix+'bias_exp']
