'''
Description: 
Date: 2023-03-11 15:51:22
LastEditTime: 2023-05-04 22:01:11
'''
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F


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
        # x_after = LH(x_LH) + HL(x_HL) + HH(x_HH)

        # ignore the hh
        x_after = LL(x_LL) + LH(x_LH) + HL(x_HL)

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
    high,low = calc_gussian(input, k=7)
    return low,high


def fft_magnitude_phase(tensor):
    # 首先将张量从NCHW格式转换为CNHW格式
    # tensor = tensor.permute(1, 0, 2, 3)
    # 进行FFT变换
    fft_tensor = torch.fft.fft2(tensor, dim=(-2, -1))
    # 计算幅值和相位
    magnitude = torch.abs(fft_tensor)
    phase = torch.angle(fft_tensor)
    return magnitude, phase

def fft_freq_filter_trasformation(input):
    # use fft to filter the image with different frequency
    # fft
    magnitude, phase = fft_magnitude_phase(input)
    # mask
    mask = torch.ones_like(magnitude)
    mask_threshold = 10
    mask[:, :, 0:mask_threshold, 0:mask_threshold] = 0
    mask[:, :, 0:mask_threshold, -mask_threshold:] = 0
    mask[:, :, -mask_threshold:, 0:mask_threshold] = 0  
    mask[:, :, -mask_threshold:, -mask_threshold:] = 0
    
    # use the mask to filter the image
    magnitude_high = magnitude * mask
    magnitude_low = magnitude * (1 - mask)

    # inverse fft
    high_freq = magnitude_high * torch.exp(1j * phase)
    low_freq = magnitude_low * torch.exp(1j * phase)

    # reconstruct the image
    high_freq = torch.fft.ifft2(high_freq, dim=(-2, -1))
    low_freq = torch.fft.ifft2(low_freq, dim=(-2, -1))

    # to real
    high_freq = torch.real(high_freq)
    low_freq = torch.real(low_freq)

    return low_freq,high_freq


def fft_magnitude_phase(signal):
    fft_x = torch.fft.fftn(signal, dim=(-2, -1))
    # 首先将张量从NCHW格式转换为CNHW格式
    # tensor = tensor.permute(1, 0, 2, 3)
    # 计算幅值和相位
    magnitude = torch.abs(fft_x)
    phase = torch.angle(fft_x)
    # 将幅值和相位连接在一起
    # fft_features = torch.cat((magnitude, phase), dim=1)
    # 将张量从CNHW格式转换回NCHW格式
    # fft_features = fft_features.permute(1, 0, 2, 3)
    # return fft_features

    return magnitude, phase

def fft_magnitude_phase_reconstruction(input_amplitude, input_phase):
    # use the amplitude and phase to reconstruct the image
    # inverse fft
    signal = input_amplitude * torch.exp(1j * input_phase)
    # reconstruct the image
    signal = torch.fft.ifft2(signal, dim=(-2, -1))
    # to real
    signal = torch.real(signal)
    return signal