'''
Description: 
Date: 2023-03-11 15:51:22
LastEditTime: 2023-03-11 23:15:46
'''
import torch
import torch.nn as nn
import numpy as np
import cv2

def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


def identity_transformation(input):
    return input

def ratio_and_bias_transformation(input):
    # masked input
    masked_input = input * 0.5 + 0.5

    return masked_input

def relu_transformation(input):
    return torch.relu(input)

def activation_mask_transformation(input):
    t=0.5
    return torch.relu(input-t)

def wavelet_transformation(input):
        """
        define some transformation
        encode -> transform -> decode
        """
        # wavelet decomposition
        LL, LH, HL, HH = get_wav(512)
        x_LL = LL(input)
        x_LH = LH(input)
        x_HL = HL(input)
        x_HH = HH(input)

        # test the function of the inverse wavelet transform
        LL, LH, HL, HH = get_wav(512, pool=False)
        # ignore the ll part
        x_after = LH(x_LH) + HL(x_HL) + HH(x_HH)

        # ignore the hh
        # x_after = LL(x_LL) + LH(x_LH) + HL(x_HL)

        # ll only
        # x_after = LL(x_LL)

        # x_after = LL(x_LL) + LH(x_LH) + HL(x_HL) + HH(x_HH)
        
        # return x_after,HH(x_LL)
        return x_after, HH(x_LL)


def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel = kernel.to(device)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel

# calculate the gaussian blur of the image
def calc_gussian(x,k):
    kernel = get_gaussian_kernel(size=k)
    padding = (kernel.shape[-1] - 1) // 2
    low_freq = gaussian_blur(x, kernel,padding=padding)
    high_freq = x - low_freq
    # output_tensor = torch.cat((low_freq, high_freq), dim=1)
    return high_freq,low_freq

def gaussian_transformation(input):
    # gaussian blur
    high,low = calc_gussian(input, k=21)
    return high