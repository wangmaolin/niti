"""
Turing Tensor Core Integer torch -> titorch -_-
Version 3 ->
decide to add a binary point to each int8 tensor
"""
import sys
import torch
import int8mm_cuda
from torch.nn.modules import Module
from torch.nn.modules import Sequential
from torch.nn import functional as F
from meters import AverageMeter
import math
import int_im2col_cuda
import numpy as np

# set CUDA_VISABLE_DEVICE when run the script to switch GPU

def PstoShiftInt32(input, shift, int8=True):
    '''
    Shift the input using our
    new pseudo stochastic rounding
    '''
    round_temp = input//Int32Tensor([2**shift])
    # stochastic rounding
    # but use the extra precision as pseudo random number
    prob = torch.abs(input - round_temp * Int32Tensor([2**shift]))
    quantized_prob = prob//Int32Tensor([2**(shift//2)])
    pseudo_rand_num = prob - quantized_prob*Int32Tensor([2**(shift//2)])

    # if shift is odd, need to make sure
    # qprob and prand have same bit width
    if shift % 2 == 1:
        pseudo_rand_num = pseudo_rand_num*Int32Tensor([2])

    round_decision = torch.where(quantized_prob <= pseudo_rand_num, Int32Tensor([0]), Int32Tensor([1]))
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)

def StoShiftInt32(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    round_temp = input//Int32Tensor([2**shift])
    prob = torch.abs(input - round_temp * Int32Tensor([2**shift]))
    rand_num = torch.randint(low = 0, high=2**int(shift),size=prob.size(), dtype = torch.int32,device=GPU)
    round_decision = torch.where(prob <= rand_num, Int32Tensor([0]), Int32Tensor([1]))
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)

def TruncateInt32(input, shift):
    round_temp = input//Int32Tensor([2**shift])
    return int8_clip(round_temp)

def RoundInt32(input, shift):
    round_temp = input//Int32Tensor([2**shift])
    prob = input - round_temp * Int32Tensor([2**shift])
    round_decision = prob//Int32Tensor([2**(shift-1)])
    return int8_clip(round_temp + round_decision)

def int8_clip(input):
    return torch.clamp(input,-127,127).type(torch.int8)

def roundoff4(size):
    return (size+3) // 4 * 4

def int8mm(lhs,rhs):
    # the cuda extension only support m,n,k as a multiply of 4
    # use torch.nn.pad to pad 0 if these dimension doesn't satisfy the
    # requirement
    m = roundoff4(lhs.size(0))
    k = roundoff4(lhs.size(1))
    n = roundoff4(rhs.size(1))

    m_diff = m - lhs.size(0)
    k_diff = k - lhs.size(1)
    n_diff = n - rhs.size(1)

    if m!=lhs.size(0) or k!=lhs.size(1):
        # faster
        A = F.pad(lhs, (0, k_diff, 0, m_diff), "constant", 0)
        # slower
        #A = torch.zeros(m,k,dtype=torch.int8).to(GPU)
        #A[:lhs.size(0),:lhs.size(1)] = lhs

        #print(lhs.is_cuda)
        #print(lhs.type())
        #ramin = A.cpu()
        #print(A.data.cpu().numpy())
        #print(A.data[0,0])
    else:
        A = lhs

    if k!=lhs.size(1) or n!=rhs.size(1):
        # faster
        B = F.pad(rhs, (0, n_diff, 0, k_diff), "constant", 0)
        # slower
        #B = torch.zeros(k,n,dtype=torch.int8).to(GPU)
        #B[:lhs.size(1),:rhs.size(1)] = rhs
    else:
        B = rhs

    temp = int8mm_cuda.int8_mm(A.contiguous(),B.contiguous())

    if m!=lhs.size(0) or n!=rhs.size(1):
        temp = temp [:lhs.size(0),:rhs.size(1)]

    return temp.contiguous()

def batch_int8mm(lhs,rhs):
    # input is rhs [batch_size,k,n], lhs [m,k]
    # output is [batch_size,m,n]
    m=lhs.size(0)
    k=lhs.size(1)
    n=rhs.size(2)
    batch_size = rhs.size(0)
    rhs_fold = rhs.transpose(0,1).contiguous()
    rhs_fold.resize_(k,batch_size*n)
    temp = int8mm(lhs.contiguous(),rhs_fold.contiguous())
    return temp.resize_(m,batch_size,n).transpose_(0,1).contiguous()

def int_unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    if input.dim() == 4:
        return int_im2col_cuda.im2col(input.contiguous(), F._pair(kernel_size),
                                   F._pair(dilation), F._pair(padding), F._pair(stride))
    else:
        raise NotImplementedError("Input Error: Only 4D input Tensors are supported (got {}D)".format(input.dim()))

def Int8Tensor(val):
    return torch.tensor(val,dtype=torch.int8).to(GPU)

def Int8Weights(size):
    temp = torch.zeros(size)
    torch.nn.init.xavier_normal_(temp)
    # torch.nn.init.xavier_uniform_(temp)
    # torch.nn.init.kaiming_uniform_(temp)
    # torch.nn.init.kaiming_normal_(temp)
    return weight_quant(temp*WEIGHT_INIT_SCALING)

def Int8zeros(size):
    return torch.zeros(size=size, dtype=torch.int8, device=GPU)

def Int32Tensor(val):
    return torch.tensor(val,dtype=torch.int32).to(GPU)

def act_calc(int32_acc,exp_in):
    '''
    calcualte the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-BITWIDTH
    if shift > 1:
        exp_out = exp_in+shift
        temp = ACT_ROUND_METHOD(int32_acc,shift)
    elif shift == 1:
        exp_out = exp_in+2
        temp = ACT_ROUND_METHOD(int32_acc,2)
    else:
        exp_out=exp_in
        temp = int32_acc.type(torch.int8)

    return temp, exp_out.type(torch.int8)

def err_calc(int32_acc):
    '''
    back propagate error
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-BITWIDTH
    if shift > 1:
        temp = ERROR_ROUND_METHOD(int32_acc,shift)
    elif shift == 1:
        temp = ERROR_ROUND_METHOD(int32_acc,2)
    else:
        temp = int32_acc.type(torch.int8)
    return temp

def grad_calc(int32_acc):
    '''
    calculate the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    if int32_bitwidth == 0:
        return Int8zeros(int32_acc.size())
    else:
        return GRAD_ROUND_METHOD(int32_acc,int32_bitwidth-GRAD_BITWIDTH)

def forward_block_float_add(a,a_exp,b,b_exp):
    # add two 8 bits block floating point tensors
    if a_exp > b_exp:
        exp_diff = a_exp - b_exp
        if exp_diff > 1:
            b=ACT_ROUND_METHOD(b.type(torch.int32),exp_diff)
            temp_exp = a_exp
        else:
            a=a.type(torch.int32)*2
            temp_exp = b_exp
    elif a_exp < b_exp:
        exp_diff = b_exp - a_exp
        if exp_diff > 1:
            a=ACT_ROUND_METHOD(a.type(torch.int32),exp_diff)
            temp_exp = b_exp
        else:
            b=b.type(torch.int32)*2
            temp_exp = a_exp
    else:
        temp_exp = a_exp

    temp = a.type(torch.int32)+b.type(torch.int32)

    if torch.max(torch.abs(temp)) > 127:
        return ACT_ROUND_METHOD(temp,2), temp_exp+2
    else:
        return temp.type(torch.int8), temp_exp

def backward_block_float_add(a,a_exp,b,b_exp):
    # add two 8 bits block floating point tensors
    if a_exp > b_exp:
        exp_diff = a_exp - b_exp
        if exp_diff > 1:
            b=ERROR_ROUND_METHOD(b.type(torch.int32),exp_diff)
            temp_exp = a_exp
        else:
            a=a.type(torch.int32)*2
            temp_exp = b_exp
    elif a_exp < b_exp:
        exp_diff = b_exp - a_exp
        if exp_diff > 1:
            a=ERROR_ROUND_METHOD(a.type(torch.int32),exp_diff)
            temp_exp = b_exp
        else:
            b=b.type(torch.int32)*2
            temp_exp = a_exp
    else:
        temp_exp = a_exp

    temp = a.type(torch.int32)+b.type(torch.int32)

    if torch.max(torch.abs(temp)) > 127:
        return ERROR_ROUND_METHOD(temp,2), temp_exp+2
    else:
        return temp.type(torch.int8), temp_exp

def unit_update_b8(var, var_grad):
    var.data = int8_clip(var.type(torch.int32)-var_grad.type(torch.int32))
    # pass

def unit_update_with_frac(var, var_grad, var_frac, frac_bits):
    var_frac -= var_grad
    # var.data = RoundInt32(var_frac,frac_bits)
    var.data = PstoShiftInt32(var_frac,frac_bits)

def parameter_decay(grad_int32acc, var):
    grad_bit_width = RangeEstimate(grad_int32acc)
    decay_ratio = int(grad_bit_width-15)
    if decay_ratio >= 0:
        grad_int32acc.data += var.type(torch.int32)*(2**decay_ratio)
    else:
        grad_int32acc.data += var.type(torch.int32)//(2**-decay_ratio)

class TiLinear(Module):
    '''
    Integer linear layer with turing GPU acceleration
    '''
    def __init__(self,in_features,out_features,bias=False,frac_bits=8):
        '''
        activation input: rhs
        weight: lhs
        activation output: results
        '''
        super(TiLinear, self).__init__()
        # initialize weight from input
        self.weight, self.weight_exp = Int8Weights(torch.Size([out_features,in_features]))
        self.frac_bits = frac_bits
        self.weight_frac = self.weight.type(torch.int32)*2**self.frac_bits
        self.momentum = Int8zeros(torch.Size([out_features,in_features])).type(torch.int32)
        if bias:
            self.bias = Int8zeros(torch.Size([out_features]))
            self.bias_exp = Int8Tensor(self.weight_exp.data.item())
            self.bias_frac = self.bias*Int8Tensor(0)

    def forward(self,input):
        '''
        GPU accelerated integer linear layer forward
        '''
        self.act_in = input
        act_in, exp_in= input
        temp = int8mm(act_in,self.weight.transpose(0,1).contiguous())

        act_out, exp_out = act_calc(temp,exp_in+self.weight_exp)

        if hasattr(self,'bias'):
            act_out,exp_out = forward_block_float_add(act_out,
                                              exp_out,
                                              self.bias,
                                              self.bias_exp)

        # save activation for backwards
        return act_out, exp_out

    def backward(self,input):
        err_in = input
        act, self.act_in_exp = self.act_in

        if hasattr(self,'bias'):
            bias_grad_int32acc = torch.sum(err_in, dim=0, dtype=torch.int32)

            if WEIGHT_DECAY:
                parameter_decay(bias_grad_int32acc, self.bias)

            self.bias_grad = grad_calc(bias_grad_int32acc)
            #update bias
            if UPDATE_WITH_FRAC:
                unit_update_with_frac(self.bias, self.bias_grad, self.bias_frac)
            elif UPDATE_WITH_MOMENTUM:
                unit_update_with_momentum(self.bias, self.bias_grad, self.bias_frac)
            else:
                unit_update_b8(self.bias, self.bias_grad)

        # propagate error to previous layer
        err_out_int32 = int8mm(err_in,self.weight)
        self.err_out = err_calc(err_out_int32)

        grad_int32acc = int8mm(err_in.transpose(0,1).contiguous(),act)

        if WEIGHT_DECAY:
            parameter_decay(grad_int32acc, self.weight)

        self.grad_int32acc=grad_int32acc
        #update weights
        if UPDATE_WITH_FRAC:
            self.grad = grad_calc_32b(grad_int32acc)
            unit_update_with_frac(self.weight, self.grad, self.weight_frac, self.frac_bits)
        else:
            self.grad = grad_calc(grad_int32acc)
            unit_update_b8(self.weight, self.grad)

        return self.err_out

class TiReLU(Module):
    '''
    Integer ReLU layer
    '''
    def forward(self, input):
        '''
        Compare the input integer tensor with its zero point
        '''
        self.act_in = input
        act_in, exp_in = input
        act_out = torch.max(act_in,Int8Tensor([0]))
        return act_out, exp_in

    def backward(self, input):
        '''
        Backward pass for ReLU
        if pluged in activation bigger than 0, propagate error input
        else propagate 0
        '''
        err_in = input
        act, _ = self.act_in
        err_out = torch.where(act > Int8Tensor([0]), err_in, Int8Tensor([0]))
        return err_out

class TiMaxpool2d(Module):
    '''
    Integer Max Pooling 2d Layer
    '''
    def __init__(self, kernel_size, stride, padding=0, dilation=1):
        super(TiMaxpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self,input):
        act_in, exp_in = input
        pool_result, self.pool_indices = F.max_pool2d(act_in.float(),
                                             self.kernel_size,
                                             self.stride,
                                             self.padding,
                                             self.dilation,
                                            return_indices=True)
        self.act_shape=act_in.shape
        return pool_result.type(torch.int8), exp_in

    def backward(self, input):
        err_in = input
        return F.max_unpool2d(err_in.type(torch.float),\
                              self.pool_indices,\
                              self.kernel_size,\
                              self.stride,
                              self.padding,
                              output_size=self.act_shape).type(torch.int8)

class TiFlat(Module):
    '''
    Flat the input integer tensor except the batch dimension
    '''
    def forward(self, input):
        self.act_in = input
        act_in, exp_in = input
        act_out = act_in.view(-1,act_in.nelement()//act_in.size(0))
        return act_out, exp_in

    def backward(self,input):
        '''
        Convert the Flat error back to the shape before flattern
        '''
        err_in = input
        act, _ = self.act_in
        return err_in.view_as(act)

def weight_quant(input):
    input_range = torch.max(torch.abs(input))
    input_bitwidth=torch.ceil(torch.log2(input_range))
    act_exp = input_bitwidth - 7
    round_val = torch.round(input/input_range*(2**7-1)).type(torch.int8).to(GPU)
    return round_val, act_exp.type(torch.int8).to(GPU)

class TiFloatToInt8(Module):
    '''
    Convert float tensors to integer tensors
    Could be used to feed in data in forward pass
    or feed in loss in backwards
    '''
    def forward(self,input):
        self.act_in = input
        input_range = torch.max(torch.abs(input))
        input_bitwidth=torch.ceil(torch.log2(input_range))
        act_exp = input_bitwidth - BITWIDTH
        round_val = torch.round(input/input_range*(2**BITWIDTH-1)).type(torch.int8).to(GPU)
        # return quantised int8 and exponent
        return round_val, act_exp.type(torch.int8).to(GPU)

def TiInt8ToFloat(input):
    '''
    Convert int tensors to float tensors
    Could be used to verify the functionality of integer NN
    '''
    act_in, exp_in = input
    return act_in.float() * (2**(exp_in.float()))

def TensorBitwidth(val):
    return torch.ceil(torch.log2(val.float()))

def RangeEstimate(input):
    '''
    Determine the activation range
    '''
    range= torch.max(torch.abs(input).float())
    if range ==0:
        return 0
    else:
        return TensorBitwidth(range)

class TiConv2d(Module):
    '''
    Integer Conv2d Layer
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 bias=False,
                 first_layer=False,
                 frac_bits=8):
        super(TiConv2d, self).__init__()
        # init integer weights from its floating point counterpart
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight, self.weight_exp= Int8Weights(torch.Size([out_channels,
                                                              in_channels,
                                                              kernel_size,
                                                              kernel_size]))

        self.frac_bits = frac_bits
        self.weight_frac = self.weight.type(torch.int32)*2**self.frac_bits

        if bias:
            self.bias = Int8zeros(torch.Size([out_channels]))
            self.bias_exp = Int8Tensor(self.weight_exp.data.item())
            self.bias_frac = Int8Tensor(0)*self.bias

        self.first_layer = first_layer

    def forward(self, input):
        # unfold input and weights
        self.act_in = input
        act_in, exp_in = input

        input_windows = int_unfold(act_in,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding)

        temp = INT8_MM(self.weight.view(self.weight.size(0), -1),input_windows)

        # fold output back
        out_size = ((act_in.shape[2] + 2 * self.padding - self.dilation
                     * (self.kernel_size-1)-1)//self.stride)+1
        temp.resize_(temp.size(0),temp.size(1),out_size,out_size)

        #save activation bitwidth for debug
        act_out,exp_out = act_calc(temp,exp_in+self.weight_exp)

        # add bias
        if hasattr(self, 'bias'):
            act_out,exp_out = forward_block_float_add(act_out,
                                              exp_out,
                                              self.bias.view(1,act_out.size(1),1,1),
                                              self.bias_exp)

        return act_out, exp_out

    def backward(self,input):
        err_in = input
        act, self.act_in_exp = self.act_in
        if hasattr(self,'bias'):
            bias_grad_int32acc = torch.sum(err_in,(0,2,3),dtype=torch.int32)

            if WEIGHT_DECAY:
                parameter_decay(bias_grad_int32acc, self.bias)

            self.bias_grad = grad_calc(bias_grad_int32acc)

            #update bias
            if UPDATE_WITH_FRAC:
                unit_update_with_frac(self.bias, self.bias_grad, self.bias_frac)
            else:
                unit_update_b8(self.bias, self.bias_grad)

        # back propagate error
        # unfold err_input

        if not self.first_layer:
            err_input_windows = int_unfold(err_in,
                                        kernel_size=self.kernel_size,
                                        stride=self.dilation,
                                        dilation=self.stride,
                                        padding=self.kernel_size-1-self.padding)
            # flip weight kernel
            weight_transposed=torch.flip(self.weight.transpose(0,1),[2,3])

            err_out_int32 = INT8_MM(
                weight_transposed.view(weight_transposed.size(0),-1),
                err_input_windows)

            # fold error output back
            err_out_int32.resize_(err_out_int32.size(0),
                                err_out_int32.size(1),
                                act.size(2),
                                act.size(3))

            self.err_out = err_calc(err_out_int32)

        # conv weight gradient
        # use err as conv kernel to do conv with act_in
        # re-range for tensor contraction
        act_input_windows = int_unfold(act.transpose(0,1),
                                     kernel_size=err_in.size(2),
                                     stride=self.stride,
                                     padding=self.padding)
        grad_int32acc = INT8_MM(
            err_in.transpose(0,1).contiguous().view(err_in.size(1),-1),
            act_input_windows)

        grad_int32acc.resize_(grad_int32acc.size(0),
                              grad_int32acc.size(1),
                              self.weight.size(2),
                              self.weight.size(3))
        grad_int32acc.transpose_(0,1)
        self.grad = grad_calc(grad_int32acc)

        #update weights
        if UPDATE_WITH_FRAC:
            self.grad = grad_calc_32b(grad_int32acc)
            unit_update_with_frac(self.weight, self.grad, self.weight_frac,self.frac_bits)
        else:
            self.grad = grad_calc(grad_int32acc)
            unit_update_b8(self.weight, self.grad)

        if not self.first_layer:
            return self.err_out

class TiConv2d_acc23(Module):
    '''
    Use fp32 Conv2d to accelerate NITI framework.
    The current naive implementation of NITI doesn't use Winograd, FFT transform
    or other GPU optimization to accelerate Conv2d,
     so it's slower than fp32 pytorch for now.
    The arithmetic still uses the same 8 bit integer.
    But the intermediated accumulation precision drops from
    32 bits to 23 bits.
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 bias=False,
                 first_layer=False,
                 frac_bits=8):
        super(TiConv2d_acc23, self).__init__()
        # init integer weights from its floating point counterpart
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight, self.weight_exp = Int8Weights(torch.Size([out_channels,
                                                              in_channels,
                                                              kernel_size,
                                                              kernel_size]))
        self.frac_bits = frac_bits
        self.weight_frac = self.weight.type(torch.int32)*2**self.frac_bits
        self.momentum = Int8zeros(torch.Size([out_channels,
                                              in_channels,
                                              kernel_size,
                                              kernel_size])).type(torch.int32)


        if bias:
            self.bias = Int8zeros(torch.Size([out_channels]))
            self.bias_exp = Int8Tensor(self.weight_exp.data.item())
            self.bias_frac = Int8Tensor(0)*self.bias

        self.first_layer = first_layer

    def forward(self, input):
        self.act_in = input
        act_in, exp_in = input
        temp = F.conv2d(input = act_in.float(),
                        weight = self.weight.float(),
                        stride = self.stride,
                        padding = self.padding).type(torch.int32)
        act_out,exp_out = act_calc(temp,exp_in+self.weight_exp)

        # add bias
        if hasattr(self, 'bias'):
            act_out,exp_out = forward_block_float_add(act_out,
                                              exp_out,
                                              self.bias.view(1,act_out.size(1),1,1),
                                              self.bias_exp)

        return act_out, exp_out

    def backward(self, input):
        err_in = input
        act, self.act_in_exp = self.act_in

        if hasattr(self,'bias'):
            bias_grad_int32acc = torch.sum(err_in,(0,2,3),dtype=torch.int32)

            if WEIGHT_DECAY:
                parameter_decay(bias_grad_int32acc, self.bias)

            self.bias_grad = grad_calc(bias_grad_int32acc)

            #update bias
            if UPDATE_WITH_FRAC:
                unit_update_with_frac(self.bias, self.bias_grad, self.bias_frac)
            else:
                unit_update_b8(self.bias, self.bias_grad)

        # first layer doesn't need to back prop error
        if not self.first_layer:
            # backward pass gradient of activation
            if self.stride != 1:
                output_padding = 1
            else:
                output_padding = 0
            err_out_int32 = F.conv_transpose2d(input=err_in.float(),
                                            weight=self.weight.float(),
                                            stride=self.stride,
                                            padding=self.padding,
                                            output_padding=output_padding).type(torch.int32)

            self.err_out = err_calc(err_out_int32)

        # calculate weight gradient
        if not self.first_layer:
            grad_int32acc = F.conv2d(input = act.transpose(0,1).float(),
                                    weight = err_in.transpose(0,1).contiguous().float(),
                                    stride=1,
                                    padding=self.padding,
                                    dilation=self.stride).type(torch.int32)

            grad_int32acc.transpose_(0,1)
            '''
            need to strip the last row due to not perfect convolution arithmetic
            '''
            if self.stride != 1:
                grad_int32acc = grad_int32acc.narrow(2,0,self.kernel_size).narrow(3,0,self.kernel_size)
        else:
            grad_int32acc = torch.nn.grad.conv2d_weight(input=act.float(),
                                                        weight_size=self.weight.shape,
                                                        grad_output=err_in.float(),
                                                        stride=self.stride,
                                                        padding=self.padding).type(torch.int32)

        if WEIGHT_DECAY:
            parameter_decay(grad_int32acc, self.weight)

        self.grad_int32acc=grad_int32acc

        #update weights
        if UPDATE_WITH_FRAC:
            self.grad = grad_calc_32b(grad_int32acc)
            unit_update_with_frac(self.weight, self.grad, self.weight_frac,self.frac_bits)
        else:
            self.grad = grad_calc(grad_int32acc)
            unit_update_b8(self.weight, self.grad)

        if not self.first_layer:
            return self.err_out

GPU='cuda:0'

BITWIDTH = 7

ACT_ROUND_METHOD = PstoShiftInt32
ERROR_ROUND_METHOD = PstoShiftInt32
GRAD_ROUND_METHOD = PstoShiftInt32

INT8_MM = batch_int8mm

WEIGHT_INIT_SCALING=1.0
