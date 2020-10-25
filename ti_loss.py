import torch
import torch.nn as nn
from torch.nn.modules import Module
from ti_torch import RangeEstimate, Int8Tensor, StoShiftInt32
from ti_torch import TiFloatToInt8, TiInt8ToFloat, GPU
from ti_torch import Int8zeros, int8_clip

class TiLoss(Module):
    def forward(self, out_val, out_exp, target):
        # err_out_exp=0
        # integer cross entropy loss
        s=out_val.type(torch.int64)
        if out_exp >-7:
            # if out_exp is big enough
            # change the base in log softmax from e to 2
            # to approx integer loss
            s=s*47274//(2**15)
            if out_exp>=0:
                s=s*2**out_exp
            else:
                s=s//(2**-out_exp)

            out_max, _ = torch.max(s,dim=1)
            offset = out_max-10
            s=s-offset.view(-1,1)
            s=torch.max(s,Int8Tensor(0).type(torch.int64))
            out_grad = 2**s-1
        else:
            # if out_exp is too small s will be all 0
            # use another apporximation 1+e^x = 1 + x + 0.5 x^2 + o(x^2)
            out_grad = 2**(1-2*out_exp.type(torch.int64)) + \
                s*2**(1-out_exp.type(torch.int64)) + s*s

        out_sum = out_grad.sum(1,dtype=torch.int64)

        out_grad = out_grad*(2**11)//out_sum.view(-1,1)
        out_grad[torch.arange(out_val.size(0)), target] -= out_grad.sum(1,dtype=torch.int64)
        self.out_grad = StoShiftInt32(out_grad.type(torch.int32),4)

        # return self.out_grad, err_out_exp
        return self.out_grad
