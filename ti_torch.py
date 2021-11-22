import torch
from torch.nn.modules import Module
import cutlassconv_cuda
import int8pool_cuda
import int8mm_cuda
import int8conv_cuda
import torch.nn.functional as F
import time
from options import args

class UpdateWeight(Module):
    """
    The parameter update part of both TiConv2d and TiLinear are the same
    Use on base class for the common function
    """
    def weight_update(self):
        p = self.weight
        """ vanilla SGD """
        self.grad, grad_shift = grad_calc(self.grad_int32acc, GRAD_BITWIDTH)
        self.grad_exp = self.err_exp + grad_shift + self.act_in_exp
        p.data = int8_clip(p.type(torch.int16)-self.grad.type(torch.int16))

def RoundShift(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    round_temp = input//(2**shift)
    prob = input - round_temp * (2**shift)
    round_decision = prob//(2**(shift-1))
    return int8_clip(round_temp + round_decision)

def StoShift(input, shift):
    '''
    Shift the input using
    stochastic rounding
    '''
    tensor_type = input.dtype
    round_temp = input//(2**shift)
    prob = torch.abs(input - round_temp * (2**shift))
    rand_num = torch.randint(low = 0, high=2**shift,size=prob.size(), dtype = tensor_type, device='cuda')
    round_decision = torch.where(prob <= rand_num,
                                 torch.tensor(0,dtype=tensor_type,device='cuda'),
                                 torch.tensor(1,dtype=tensor_type,device='cuda'))
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)

def PstoShift(input, shift):
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
                                 torch.tensor(0,dtype=torch.int32,device='cuda'),
                                 torch.tensor(1,dtype=torch.int32,device='cuda'))
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)

def int8_clip(input, clip_val=127):
    return torch.clamp(input,-clip_val, clip_val).type(torch.int8)

def Int8Tensor(val):
    return torch.tensor(val,dtype=torch.int8).to(GPU)

def UniformWeights(size):
    if len(size) == 4:
        temp = torch.zeros(size).permute(0,3,1,2).contiguous()
        torch.nn.init.xavier_uniform_(temp)
        return weight_quant(temp.permute(0,2,3,1).contiguous())
    else:
        temp = torch.zeros(size)
        torch.nn.init.xavier_uniform_(temp)
        return weight_quant(temp)

def NormalWeight(size):
    if len(size) == 4:
        temp = torch.zeros(size).permute(0,3,1,2).contiguous()
        torch.nn.init.xavier_normal_(temp)
        return weight_quant(temp.permute(0,2,3,1).contiguous())
    else:
        temp = torch.zeros(size)
        torch.nn.init.xavier_normal_(temp)
        return weight_quant(temp)

def Int8zeros(size):
    return torch.zeros(size=size, dtype=torch.int8, device=GPU)

def Int8Tensor(val):
    return torch.tensor(val,dtype=torch.int8).to(GPU)

def act_calc(int32_acc, exp_in):
    '''
    calcualte the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-BITWIDTH
    if shift > 0:
        exp_out = exp_in+shift
        temp = ACT_ROUND_METHOD(int32_acc, shift)
    else:
        exp_out=exp_in
        temp = int32_acc.type(torch.int8)

    return temp, exp_out

def err_calc(int32_acc):
    '''
    calcualte the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-BITWIDTH
    if shift > 0:
        temp =ERROR_ROUND_METHOD(int32_acc, shift)
        exp_out = shift
    else:
        temp = int32_acc.type(torch.int8)
        exp_out= 0

    return temp, exp_out

def grad_calc(int32_acc, mu):
    '''
    calculate the exponent value of accumulation results
    when shifting the int32 back to int8
    '''
    int32_bitwidth = RangeEstimate(int32_acc)
    shift = int32_bitwidth-mu
    if int32_bitwidth == 0:
        return Int8zeros(int32_acc.size()), 0
    elif shift < 1:
        return int32_acc.type(torch.int8), 0
    else:
        return GRAD_ROUND_METHOD(int32_acc,int32_bitwidth-mu), shift

def roundoff4(size):
    return (size+3) // 4 * 4

def int8mm(lhs,rhs): # the cuda extension only support n,k as a multiply of 4
    # use torch.nn.pad to pad 0 if these dimension doesn't satisfy the
    # requirement
    k = roundoff4(lhs.size(1))
    n = roundoff4(rhs.size(1))

    k_diff = k - lhs.size(1)
    n_diff = n - rhs.size(1)

    if  k!=lhs.size(1):
        A = F.pad(lhs, (0, k_diff, 0, 0), "constant", 0)
    else:
        A = lhs

    if k!=lhs.size(1) or n!=rhs.size(1):
        B = F.pad(rhs, (0, n_diff, 0, k_diff), "constant", 0)
    else:
        B = rhs

    temp = int8mm_cuda.int8_mm(A, B)

    if n!=rhs.size(1):
        temp = temp [:lhs.size(0),:rhs.size(1)]

    return temp.contiguous()

def conv2d_int8(input, weight, stride=1, padding=1):
    """ only input channel(tensor.size(3)) is a multiple of 16"""
    if input.size(3) % 16 != 0:
        padding_channels = 16 - input.size(3) % 16
        input_padded = F.pad(input, (0, padding_channels),"constant", 0)
        weight_padded = F.pad(weight, (0, padding_channels),"constant", 0)
    else:
        input_padded = input
        weight_padded = weight

    if weight.size(1) <= 32:
        temp = cutlassconv_cuda.sp_conv_optimized(input_padded, weight_padded, stride, padding)
    else:
        temp = cutlassconv_cuda.sp_conv(input_padded, weight_padded, stride, padding)
    return temp

def conv2d_weight_int8(input, weight_size, grad_output, stride=1, padding=0, dilation=1):
    in_channels = input.shape[3]
    out_channels = grad_output.shape[3]
    min_batch = input.shape[0]

    grad_output = grad_output.permute(0,3,1,2)
    grad_output = grad_output.contiguous().repeat(1, in_channels, 1, 1)
    grad_output = grad_output.contiguous().reshape(grad_output.shape[0] * grad_output.shape[1],
                                                1,
                                                grad_output.shape[2],
                                                grad_output.shape[3])
    grad_output = grad_output.permute(0,2,3,1).contiguous()

    input = input.permute(0,3,1,2)
    input= input.contiguous().reshape(1,
                                   input.shape[0] * input.shape[1],
                                   input.shape[2],
                                   input.shape[3])
    input = input.permute(0,2,3,1).contiguous()

    grad_weight = int8conv_cuda.group_conv(input,
                                           grad_output,
                                           dilation,
                                           padding,
                                           stride,
                                           in_channels * min_batch)

    grad_weight = grad_weight.permute(0,3,1,2)

    grad_weight = grad_weight.view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
        grad_weight.shape[3])

    grad_weight = grad_weight.sum(dim=0).view(
       in_channels, out_channels,
       grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
           2, 0, weight_size[1]).narrow(3, 0, weight_size[2])
    grad_weight = grad_weight.permute(0,2,3,1).contiguous()

    return grad_weight

class TiLinear(UpdateWeight):
    def __init__(self,in_features,out_features, last_layer = False):
        super(TiLinear, self).__init__()
        self.weight, self.weight_exp = WEIGHT_INIT_METHOD(torch.Size([out_features,in_features]))
        self.last_layer = last_layer

    def forward(self,input):
        """ save activation for backwards """
        self.act_in = input
        act_in, exp_in = input

        temp = int8mm(act_in, self.weight.transpose(0,1).contiguous())

        # if self.last_layer:
            # act_out, exp_out = temp, exp_in+self.weight_exp
        # else:
        act_out, exp_out = act_calc(temp,exp_in+self.weight_exp)

        return act_out, exp_out

    def backward(self, input):
        err_in, self.err_exp = input
        act, self.act_in_exp = self.act_in

        err_out_int32 = int8mm(err_in, self.weight)
        err_out, shift_bits = err_calc(err_out_int32)
        self.err_exp += (shift_bits + self.weight_exp)

        self.grad_int32acc = int8mm(err_in.transpose(0,1).contiguous(), act)
        self.weight_update()

        return err_out, self.err_exp

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
        err_in, err_exp = input
        act, _ = self.act_in
        err_out = torch.where(act > Int8Tensor([0]), err_in, Int8Tensor([0]))
        return err_out, err_exp

class TiMaxpool2d(Module):
    '''
    Integer Max Pooling 2d Layer
    '''
    def __init__(self, kernel_size, stride, padding=0):
        super(TiMaxpool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self,input):
        self.act_in, exp_in = input

        self.act_out = int8pool_cuda.int8_pool(self.act_in, self.kernel_size, self.stride, self.padding)

        return self.act_out, exp_in

    def backward(self, input):
        """ cudnn pool backward function doesn't support int8, use half instead"""
        err_in, err_exp = input

        err_out = int8pool_cuda.int8_pool_backward(self.act_out.half(), err_in.half(), self.act_in.half(),
                                              self.kernel_size, self.stride, self.padding).type(torch.int8)

        return err_out, err_exp

class TiFlat(Module):
    ''' Flat the input integer tensor except the batch dimension '''
    def forward(self, input):
        self.act_in = input
        act_in, exp_in = input
        act_out = act_in.view(-1,act_in.nelement()//act_in.size(0))
        return act_out, exp_in

    def backward(self,input):
        '''
        Convert the Flat error back to the shape before flattern
        '''
        err_in, err_exp = input
        act, _ = self.act_in
        return err_in.view_as(act), err_exp

def weight_quant(input):
    input_range = torch.max(torch.abs(input))
    input_bitwidth=torch.ceil(torch.log2(input_range))
    act_exp = input_bitwidth - 7
    round_val = torch.round(input/input_range*(2**7-1)).type(torch.int8).to(GPU)
    return round_val, act_exp

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

def TensorBitwidth(val):
    return int(torch.ceil(torch.log2(val.float())))

def RangeEstimate(input):
    '''
    Determine the activation range
    '''
    range= torch.max(torch.abs(input).float())
    if range ==0:
        return 0
    else:
        return TensorBitwidth(range)

class TiConv2d(UpdateWeight):
    '''
    conv 3x3 with dilation 1, stride 1, padding 1
    NHWC format
    '''
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride =1, padding =1, first_layer=False):
        super(TiConv2d, self).__init__()
        self.weight, self.weight_exp = WEIGHT_INIT_METHOD(torch.Size([out_channels, kernel_size, kernel_size, in_channels]))
        self.first_layer = first_layer
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        self.act, self.act_in_exp = input

        temp = conv2d_int8(self.act, self.weight, self.stride, self.padding)
        act_out, exp_out = act_calc(temp, self.act_in_exp + self.weight_exp)

        return act_out, exp_out

    def backward(self, input):
        err_in, self.err_exp = input

        """ first layer doesn't need to back prop error """
        if not self.first_layer:
            weight_transposed = torch.flip(self.weight,[1,2]).transpose(0,3).contiguous()
            err_out_int32 = conv2d_int8(err_in, weight_transposed, stride=self.stride, padding=self.padding)
            err_out, shift_bits = err_calc(err_out_int32)
            self.err_exp += (shift_bits + self.weight_exp)

        """ calculate weight gradient
        the first layer use a group conv trick to accelerate like the pytorch code in torch.nn.grad.conv2d_weight
        """
        if self.first_layer:
            if self.act.size(0) % 4 == 0 and self.stride == 1:
                self.grad_int32acc = conv2d_weight_int8(input = self.act,
                                                        weight_size = self.weight.shape,
                                                        grad_output = err_in,
                                                        stride = self.stride,
                                                        padding = self.padding)
            else:
                """ the last batch may have odd size, current cuda extension doesn't support it"""
                self.grad_int32acc = torch.nn.grad.conv2d_weight(input = self.act.permute(0,3,1,2).contiguous().float(),
                                                                 weight_size = torch.Size([self.weight.size(0),
                                                                                           self.weight.size(3),
                                                                                           self.weight.size(1),
                                                                                           self.weight.size(2)]),
                                                                 grad_output = err_in.permute(0,3,1,2).contiguous().float(),
                                                                 stride = self.stride,
                                                                 padding = self.padding).permute(0,2,3,1).contiguous()
        else:
            act_transposed = self.act.transpose(0,3).contiguous()
            err_in_transposed = err_in.transpose(0,3).contiguous()
            self.grad_int32acc = conv2d_int8(act_transposed, err_in_transposed, stride=self.stride, padding=self.padding).transpose(0,3).contiguous()

        self.weight_update()

        if not self.first_layer:
            return err_out, self.err_exp

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
        err_in, err_exp = input
        err_out = err_in*self.drop_mask
        return err_out, err_exp

GPU='cuda'
BITWIDTH = 7

ACT_ROUND_METHOD = RoundShift
ERROR_ROUND_METHOD = RoundShift

GRAD_ROUND_METHOD = PstoShift
WEIGHT_INIT_METHOD = UniformWeights