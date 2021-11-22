import torch
from torch.nn.modules import Module
from ti_torch import RangeEstimate
from ti_torch import TiFloatToInt8
import collections
from ti_loss import TiLoss

class TiNet(Module):
    '''
    Training deep integer network
    with learning rate schedule
    '''
    def __init__(self):
        super(TiNet, self).__init__()
        self.loss = TiLoss()

    def forward(self, x):
        x_data, x_exp = TiFloatToInt8(x)
        x_data = x_data.permute(0,2,3,1).contiguous()
        x = x_data, x_exp
        self.out, self.out_exp = self.forward_layers(x)
        self.out_bits = RangeEstimate(self.out)
        return self.out, self.out_exp

    def backward(self, target):
        x = self.loss(self.out, self.out_exp, target)
        for layer in reversed(self.forward_layers):
            if isinstance(layer,torch.nn.Sequential):
                for block in reversed(layer):
                    x=block.backward(x)
            else:
                x=layer.backward(x)

    def state_dict(self):
        state_dict=collections.OrderedDict()
        for idx,l in enumerate(self.forward_layers):
            if hasattr(l,'weight'):
                layer_prefix = 'layers.'+str(idx)+'.'
                state_dict[layer_prefix+'weight']=l.weight
                state_dict[layer_prefix+'weight_exp']=l.weight_exp
        return state_dict

    def load_state_dict(self,state_dict):
        for idx,l in enumerate(self.forward_layers):
            if hasattr(l,'weight'):
                layer_prefix = 'layers.'+str(idx)+'.'
                l.weight.data=state_dict[layer_prefix+'weight']
                l.weight_exp.data=state_dict[layer_prefix+'weight_exp']
