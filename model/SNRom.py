import torch
import torch.nn as nn
import torch.nn.functional as F
from model.octconv import *
import model.venconv as venconv
from model.median_pooling import median_pool_2d


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, alpha_in=0.5,
                 alpha_out=0.5, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ResBlock, self).__init__()

        self.conv1 = Conv_BN_ACT(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.PReLU)
        self.conv2 = Conv_BN_ACT(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.PReLU)

    def forward(self, x):
        x_hi, x_li = x

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        x_h = x_h + x_hi
        x_l = x_l + x_li

        return x_h, x_l


class SNRom(nn.Module):
    def __init__(self, in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                 alpha_out=0.5, stride=1, padding=0, dilation=1, groups=1, bias=False, hide_layers=8):
        super(SNRom, self).__init__()
        # parameters
        self.kernel_size = kernel_size
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.hide_layers = hide_layers
        # convolution layers
        self.octavecons = nn.ModuleList(
            Conv_BN_ACT(hide_channels, hide_channels, kernel_size, alpha_in, alpha_out, stride,
                        padding, dilation, groups, bias, activation_layer=nn.ReLU) for i in range(hide_layers * 2))
        # residual layers
        self.rescons = nn.ModuleList(ResBlock(hide_channels, hide_channels, kernel_size, alpha_in,
                                              alpha_out, stride, padding, dilation, groups) for i in
                                     range(hide_layers * 2))
        # for input
        self.downs = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.proninl = venconv.Conv_BN_ACT(int(in_channels), int(alpha_in * hide_channels),
                              kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.PReLU
        )
        self.proninh = venconv.Conv_BN_ACT(int(in_channels), hide_channels - int(alpha_in * hide_channels),
                                           kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.PReLU
        )

        # for output
        self.proutl = venconv.Conv_BN_ACT(int(alpha_in * hide_channels), out_channels,
                                           kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.ReLU
        )
        self.prouth = venconv.Conv_BN_ACT(hide_channels - int(alpha_in * hide_channels), out_channels,
                                           kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.ReLU
        )

        self.proutc = venconv.Conv_BN_ACT(out_channels, out_channels,
                                          kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.ReLU
        )

    def forward(self, x):

        x_l = self.downs(x)
        x_l = self.proninl(x_l)
        x_h = self.proninh(x)

        for i in range(self.hide_layers * 2):
            x_h, x_l = self.octavecons[i]((x_h, x_l))

            if i < self.hide_layers:
                x_h = median_pool_2d(x_h, self.kernel_size, self.stride, self.padding, self.dilation)
                x_l = median_pool_2d(x_l, self.kernel_size, self.stride, self.padding, self.dilation)

            x_h, x_l = self.rescons[i]((x_h, x_l))

        x_h = self.prouth(x_h)

        output = self.proutc(x_h)

        x_l = self.proutl(x_l)

        return output, x_l
