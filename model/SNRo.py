import torch
import torch.nn as nn
import torch.nn.functional as F
from model.octconv import *


class SNRo(nn.Module):
    def __init__(self, in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                 alpha_out=0.5, stride=1, padding=0, dilation=1, groups=1, bias=False, hide_layers=8):
        super(SNRo, self).__init__()
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
        # layers
        self.octavecons = nn.ModuleList(
            Conv_BN_ACT(hide_channels, hide_channels, kernel_size, alpha_in, alpha_out, stride,
                        padding, dilation, groups, bias, activation_layer=nn.ReLU) for i in range(hide_layers * 2))

        self.rescons = nn.ModuleList(ResBlock(hide_channels, hide_channels, kernel_size, alpha_in,
                                              alpha_out, stride, padding, dilation, groups) for i in
                                     range(hide_layers * 2))
        # for input
        self.downs = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.proninl = nn.Conv2d(int(in_channels), int(alpha_in * hide_channels),
                              kernel_size, 1, padding, dilation, groups, bias)
        self.proninh = nn.Conv2d(int(in_channels), hide_channels - int(alpha_in * hide_channels),
                                 kernel_size, 1, padding, dilation, groups, bias)
        # for output
        self.prouth = nn.Conv2d(hide_channels - int(alpha_in * hide_channels), out_channels,
                                kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.proutc = nn.Conv2d(out_channels, out_channels,
                                 kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        #activate layer
        self.actPReLU = nn.PReLU()
        self.actReLU = nn.ReLU()

    def forward(self, x):

        x_l = self.downs(x)
        x_l = self.actPReLU(self.proninl(x_l))
        x_h = self.actPReLU(self.proninh(x))

        for i in range(self.hide_layers * 2):
            x_h, x_l = self.octavecons[i]((x_h, x_l))
            x_h, x_l = self.rescons[i]((x_h, x_l))

        x_h = self.prouth(x_h)
        x_h = self.actReLU(x_h)

        output = self.proutc(x_h)
        output = self.actReLU(output)

        return output
