# NITI: Training Integer Neural Networks Using Integer-only Arithmetic

## Introduction
NITI is a training framework which uses 8 bits signed integer exclusively to train neural network.

## Prerequist
### Hardware 
Nvidia GPU with tensor cores like V100, GTX 2080Ti, T4
### Environment
Ubuntu 18.04 with pytorch(python3 version),bokeh installed

### install int8 image to column cuda extension under pytorch folder
cd pytorch/int8im2col-extension
sudo python3 setup.py install

### install int8 matrix multiply cuda extension under pytorch folder
cd pytorch/tcint8mm-extension
sudo python3 setup.py install

## Training int8 VGG on CIFAR10 Example
./train_vgg_cifar10.sh

## key code
### ti_torch.py 
Implementation of convolution, fully connected layer with int8 forward pass and backward pass 

### pytorch/tcint8mm-extension
CUDA extension using tensor core to accelerate 8 bit signed integer matrix multiply