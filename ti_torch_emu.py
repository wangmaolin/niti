import sys
import torch
from torch.nn.modules import Module
from torch.nn.modules import Sequential
from torch.nn import functional as F
from meters import AverageMeter
import math
import numpy as np
import int8mm_cuda
import int8conv_cuda
# from torch.utils.cpp_extension import load

# conv2d_cudnn = load(name="conv2d_backward", sources=["conv2d_backward.cpp"], verbose=True)

def RoundShift(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    round_temp = input//(2**shift)
    prob = torch.abs(input - round_temp * (2**shift))
    round_decision = prob//(2**(shift-1))
    return int8_clip(round_temp + round_decision)

def StoShift(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    return input.type(torch.int8)
    tensor_type = input.dtype
    round_temp = input//(2**shift)
    prob = torch.abs(input - round_temp * (2**shift))
    rand_num = torch.randint(low = 0, high=2**shift,size=prob.size(), dtype = tensor_type, device='cuda')
    round_decision = torch.where(prob <= rand_num,
                                 torch.tensor(0,dtype=tensor_type,device='cuda'),
                                 torch.tensor(1,dtype=tensor_type,device='cuda'))
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)

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

def RoundInt32(input, shift):
    round_temp = input//Int32Tensor([2**shift])
    prob = input - round_temp * Int32Tensor([2**shift])
    round_decision = prob//Int32Tensor([2**(shift-1)])
    return int8_clip(round_temp + round_decision)

def RoundFloat(input, shift):
    with torch.no_grad():
        round_temp = input//2**shift
        prob = input - round_temp * (2**shift)
        round_decision = prob//(2**(shift-1))
        return int8_clip(round_temp + round_decision)

def StoShiftFloat(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    round_temp = input//(2**shift)
    prob = torch.abs(input - round_temp * (2**shift))
    rand_num = torch.randint(low = 0, high=2**int(shift),size=prob.size(), dtype=torch.float, device='cuda')
    round_decision = torch.where(prob <= rand_num,
                                 torch.cuda.FloatTensor([0]),
                                 torch.cuda.FloatTensor([1]))
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)

def PstoShiftFloat(input, shift):
    '''
    Shift the input using our
    new pseudo stochastic rounding
    '''
    round_temp = input//(2**shift)
    # stochastic rounding
    # but use the extra precision as pseudo random number
    prob = torch.abs(input - round_temp * (2**shift))
    quantized_prob = prob//(2**(shift//2))
    pseudo_rand_num = prob - quantized_prob*(2**(shift//2))

    # if shift is odd, need to make sure
    # qprob and prand have same bit width
    if shift % 2 == 1:
        pseudo_rand_num = pseudo_rand_num*2

    round_decision = torch.where(quantized_prob <= pseudo_rand_num,
                                 torch.cuda.FloatTensor([0]),
                                 torch.cuda.FloatTensor([1]))
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)

def int8_clip(input):
    return torch.clamp(input,-127,127).type(torch.int8)

def Int8Tensor(val):
    return torch.tensor(val,dtype=torch.int8).to(GPU)

def NormalWeights(size):
    temp = torch.zeros(size)
    torch.nn.init.xavier_normal_(temp)
    return weight_quant(temp)

def UniformWeights(size):
    temp = torch.zeros(size)
    torch.nn.init.xavier_uniform_(temp)
    return weight_quant(temp)

def BimodalWeights(size):
    temp = torch.zeros(size)
    torch.nn.init.xavier_normal_(temp)
    range = torch.max(torch.abs(temp))
    temp.data = temp - torch.sign(temp)*range
    return weight_quant(temp*0.3)

def Int8zeros(size):
    return torch.zeros(size=size, dtype=torch.int8, device=GPU)

def Int8Tensor(val):
    return torch.tensor(val,dtype=torch.int8).to(GPU)

def Int32Tensor(val):
    return torch.tensor(val,dtype=torch.int32).to(GPU)

def act_calc(int32_acc,exp_in):
    '''
    calcualte the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-BITWIDTH
    if shift > 0:
        exp_out = exp_in+shift
        temp = ACT_ROUND_METHOD(int32_acc,shift)
    else:
        exp_out=exp_in
        temp = int32_acc.type(torch.int8)

    return temp, exp_out.type(torch.int16)

def act_calc_last_layer(int32_acc,exp_in):
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-LAST_LAYER_BITWIDTH
    if shift > 0:
        exp_out = exp_in+shift
        round_temp = int32_acc//2**shift
        prob = int32_acc - round_temp * (2**shift)
        round_decision = prob//(2**(shift-1))
        temp = round_temp + round_decision
    else:
        exp_out=exp_in
        temp = int32_acc

    return temp, exp_out.type(torch.int16)

def err_calc(int32_acc):
    '''
    calcualte the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-BITWIDTH
    if shift > 0:
        temp = ERROR_ROUND_METHOD(int32_acc,shift)
        exp_out = shift.type(torch.int8)
    # elif shift == 1:
        # temp = ERROR_ROUND_METHOD(int32_acc,2)
        # exp_out = Int8Tensor(2)
    else:
        temp = int32_acc.type(torch.int8)
        exp_out=Int8Tensor(0)

    return temp, exp_out

def grad_calc(int32_acc, mu):
    '''
    calculate the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-mu
    if int32_bitwidth == 0:
        return Int8zeros(int32_acc.size())
    elif shift < 1:
        return int32_acc.type(torch.int8)
    else:
        return GRAD_ROUND_METHOD(int32_acc,int32_bitwidth-mu)

def unit_update(var, var_grad, var_momentum = None):
    if MOMENTUM > 0 or var_momentum is not None:
        var_momentum.data = int8_clip(var_momentum.type(torch.int16)*MOMENTUM//16 +\
                                      var_grad.type(torch.int16))
                                      # var_grad.type(torch.int16)*(16-MOMENTUM)//16)

        var.data = int8_clip(var.type(torch.int16)-var_momentum.type(torch.int16))
    else:
        var.data = int8_clip(var.type(torch.int16)-var_grad.type(torch.int16))

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
        A = F.pad(lhs, (0, k_diff, 0, m_diff), "constant", 0)
    else:
        A = lhs

    if k!=lhs.size(1) or n!=rhs.size(1):
        B = F.pad(rhs, (0, n_diff, 0, k_diff), "constant", 0)
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

def int8mm_acc23(a,b):
    with torch.no_grad():
        temp = torch.matmul(a.float(),b.float())
    # return temp.type(torch.int32)
    return temp

# def int8mm_acc11(a,b):
    # temp = torch.matmul(a.half()/2.0**4,b.half()/2.0**4).float()*2.0**8
    # return temp.type(torch.int32)

# def conv2d_acc11(input, weight, stride, padding):
    # temp = F.conv2d(input = input.half()/2.0**4,
                    # weight = weight.half()/2.0**4,
                    # stride = stride,
                    # padding = padding).float()*2.0**8
    # return temp.type(torch.int32)

def conv2d_acc23(input, weight, stride, padding, dilation):
    temp = F.conv2d(input = input.float(),
                    weight = weight.float(),
                    stride = stride,
                    padding = padding,
                    dilation = dilation)
    return temp

def conv2d_int8(input, weight, stride, padding, dilation):
    temp = int8conv_cuda.int8_conv(input.permute(0,2,3,1).contiguous(),
                                   weight.permute(0,2,3,1).contiguous(),
                                   stride,
                                   padding,
                                   dilation)
    return temp.permute(0,3,1,2).contiguous()

def conv2d_transposed_acc23(input, weight, stride, padding, output_padding):
    with torch.no_grad():
        temp = F.conv_transpose2d(input=input.float(),
                                  weight=weight.float(),
                                  stride=stride,
                                  padding=padding,
                                  output_padding=output_padding)
        return temp
    # return temp.type(torch.int32)

def conv2d_weightgrad_acc23(input, weight_size, grad_output, stride, padding):
    """
    Becareful while using torch.nn.grad.conv2d_weight,
    This function is slow and consume huge GPU memory.
    """
    with torch.no_grad():
        temp = torch.nn.grad.conv2d_weight(input.float(),
                                           weight_size,
                                           grad_output.float(),
                                           stride,
                                           padding)
        # temp = conv2d_cudnn.backward(list(weight_size),
                                     # grad_output.float(),
                                     # input.float(),
                                     # int(padding),
                                     # int(stride),
                                     # int(1),
                                     # int(1),
                                     # True,
                                     # False)
        return temp
    # return temp.type(torch.int32)

class TiLinear_acc23(Module):
    '''
    Integer linear layer with turing GPU acceleration
    '''
    def __init__(self,in_features,out_features, last_layer = False):
        '''
        activation input: rhs
        weight: lhs
        activation output: results
        '''
        super(TiLinear_acc23, self).__init__()
        # initialize weight from input
        self.weight, self.weight_exp = WEIGHT_INIT_METHOD(torch.Size([out_features,in_features]))
        self.last_layer = last_layer
        # self.momentum = Int8zeros(torch.Size([out_features,in_features]))

    def forward(self,input):
        '''
        GPU accelerated integer linear layer forward
        '''
        self.act_in = input
        act_in, exp_in= input

        temp = INT8_GEMM(act_in, self.weight.transpose(0,1).contiguous())

        if self.last_layer:
            # act_out, exp_out = temp, exp_in+self.weight_exp
            act_out, exp_out = act_calc_last_layer(temp, exp_in+self.weight_exp)
        else:
            act_out, exp_out = act_calc(temp,exp_in+self.weight_exp)

        # save activation for backwards
        return act_out, exp_out

    def backward(self,input):
        err_in = input
        act, self.act_in_exp = self.act_in

        # propagate error to previous layer
        err_out_int32 = INT8_GEMM(err_in, self.weight)
        err_out, _ = err_calc(err_out_int32)

        grad_int32acc = INT8_GEMM(err_in.transpose(0,1).contiguous(), act)

        #update weights
        grad = grad_calc(grad_int32acc, GRAD_BITWIDTH)

        unit_update(self.weight, grad)

        return err_out

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
        # pool_result, self.pool_indices = F.max_pool2d(act_in.float(),
        pool_result, self.pool_indices = F.max_pool2d(act_in.half(),
                                                      self.kernel_size,
                                                      self.stride,
                                                      self.padding,
                                                      self.dilation,
                                                      return_indices=True)
        self.act_shape=act_in.shape
        return pool_result.type(torch.int8), exp_in

    def backward(self, input):
        err_in = input
        # return F.max_unpool2d(err_in.type(torch.float),\
        return F.max_unpool2d(err_in.half(),\
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

class TiBiasAdd(Module):
    def __init__(self, out_channels):
        super(TiBiasAdd, self).__init__()
        """ this is actuall bias, but use weight is convenient for visualization related code"""
        self.weight = Int8zeros(torch.Size([out_channels]))
        self.weight_exp = None

    def forward(self,input):
        act_in, exp_in = input
        if self.weight_exp is None:
            self.weight_exp = exp_in
            return act_in, exp_in
        else:
            exp_diff = exp_in - self.weight_exp
            if exp_diff>0:
                aligned_bias = RoundInt32(self.weight.type(torch.int32), exp_diff)
            elif exp_diff<0:
                aligned_bias = int8_clip(self.weight.type(torch.int32)*2**exp_diff)
            else:
                aligned_bias = self.weight

            if TRAINING:
                self.weight_exp = exp_in
                self.weight.data = aligned_bias

            if len(act_in.shape) == 4:
                temp = aligned_bias.type(torch.int32).view(1,act_in.size(1),1,1) + act_in.type(torch.int32)

            else:
                temp = aligned_bias.type(torch.int32) + act_in.type(torch.int32)

            act_out, exp_out = act_calc(temp,exp_in)
            return act_out, exp_out

    def backward(self,input):
        err_in = input
        if len(err_in.shape) == 4:
            grad_int32acc = torch.sum(err_in,(0,2,3),dtype=torch.int32)
        else:
            grad_int32acc = torch.sum(err_in, dim=0, dtype=torch.int32)

        grad = grad_calc(grad_int32acc, 1)

        unit_update(self.weight, grad)

        return err_in

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
                 first_layer=False):
        super(TiConv2d_acc23, self).__init__()
        # init integer weights from its floating point counterpart
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight, self.weight_exp = WEIGHT_INIT_METHOD(torch.Size([out_channels,
                                                                  in_channels,
                                                                  kernel_size,
                                                                  kernel_size]))
        # self.momentum = Int8zeros(torch.Size([out_channels,
                                              # in_channels,
                                              # kernel_size,
                                              # kernel_size]))

        self.first_layer = first_layer

    def forward(self, input):
        self.act_in = input
        act_in, exp_in = input
        if self.first_layer:
            temp = INT8_CONV2D_FIRST(input = act_in,
                                     weight = self.weight,
                                     stride = self.stride,
                                     padding = self.padding,
                                     dilation = self.dilation)
        else:
            temp = INT8_CONV2D(input = act_in,
                               weight = self.weight,
                               stride = self.stride,
                               padding = self.padding,
                               dilation = self.dilation)

        act_out,exp_out = act_calc(temp,exp_in+self.weight_exp)

        return act_out, exp_out

    def backward(self, input):
        err_in = input
        act, self.act_in_exp = self.act_in

        # first layer doesn't need to back prop error
        if not self.first_layer:
            # backward pass gradient of activation
            if self.stride != 1:
                output_padding = 1
            else:
                output_padding = 0

            err_out_int32 = conv2d_transposed_acc23(input=err_in,
                                                    weight=self.weight,
                                                    stride=self.stride,
                                                    padding=self.padding,
                                                    output_padding=output_padding)

            err_out, self.err_exp = err_calc(err_out_int32)

        # calculate weight gradient
        # grad_int32acc = conv2d_acc23(input = act.transpose(0,1),
        grad_int32acc = INT8_CONV2D(input = act.transpose(0,1).contiguous(),
                                    weight = err_in.transpose(0,1).contiguous(),
                                    stride=1,
                                    padding=self.padding,
                                    dilation=self.stride)
        grad_int32acc.transpose_(0,1)

        """ need to strip the last row due to not perfect convolution arithmetic """
        if self.stride != 1:
            grad_int32acc = grad_int32acc.narrow(2,0,self.kernel_size).narrow(3,0,self.kernel_size)

        grad = grad_calc(grad_int32acc, GRAD_BITWIDTH)

        unit_update(self.weight, grad)

        if not self.first_layer:
            return err_out

class TiDropout(Module):
    '''
    Integer Dropout layer
    '''
    def forward(self, input):
        if TRAINING:
            act_in, exp_in = input
            self.drop_mask = torch.randint(low=0, high=2, size=(act_in.size(1),),dtype=torch.int8,device=GPU)
            act_out = act_in*self.drop_mask
            return act_out, exp_in+1
        else:
            return input

    def backward(self, input):
        err_in = input
        err_out = err_in*self.drop_mask
        return err_out

class alex_first_conv(TiConv2d_acc23):
    """
    The first conv layer of Alexnet is special due to its weird convolution set up
    which leads to not exact conv arithmetic
    """
    def backward(self, input):
        err_in = input
        act, self.act_in_exp = self.act_in

        grad_int32acc = conv2d_weightgrad_acc23(input=act,
                                                weight_size=self.weight.shape,
                                                grad_output=err_in,
                                                stride=self.stride,
                                                padding=self.padding)

        grad = grad_calc(grad_int32acc, GRAD_BITWIDTH)

        if self.training:
            unit_update(self.weight, grad)

GPU='cuda'
BITWIDTH = 7
ACT_ROUND_METHOD = RoundShift
ERROR_ROUND_METHOD = StoShift
GRAD_ROUND_METHOD = StoShift

# INT8_CONV2D = conv2d_int8
INT8_CONV2D = conv2d_acc23
INT8_CONV2D_FIRST = conv2d_acc23
INT8_GEMM = int8mm_acc23

WEIGHT_INIT_METHOD = UniformWeights
