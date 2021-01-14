import torch
import torch.nn as nn
import torch.nn.functional as F
from model.venconv import *


class SNRc(nn.Module):
    def __init__(self, in_channels=3, hide_channels=64, out_channels=1, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, hide_layers=8):
        super(SNRc, self).__init__()
        # parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.hide_layers = hide_layers
        # layers
        self.convs = nn.ModuleList(
            Conv_BN_ACT(hide_channels, hide_channels, kernel_size, stride,
                        padding, dilation, groups, bias, activation_layer=nn.ReLU) for i in range(hide_layers * 2))

        self.rescons = nn.ModuleList(ResBlock(hide_channels, hide_channels, kernel_size,
                                              stride, padding, dilation, groups) for i in range(hide_layers * 2))
        # for input
        self.proin = Conv_BN_ACT(in_channels, hide_channels, kernel_size, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.PReLU)
        # for output
        self.prout1 = Conv_BN_ACT(hide_channels, hide_channels, kernel_size, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)
        
        self.prout2 = Conv_BN_ACT(hide_channels, out_channels, kernel_size, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)

    def forward(self, x):

        x = self.proin(x)

        for i in range(self.hide_layers * 2):
            x = self.convs[i](x)
            x = self.rescons[i](x)

        output = self.prout1(x)
        output = self.prout2(output)

        return output