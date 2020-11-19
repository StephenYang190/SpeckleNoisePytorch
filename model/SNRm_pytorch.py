import torch
import torch.nn as nn
import torch.nn.functional as F
from model.octconv import *


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


def unpack_param_2d(param):
    try:
        p_H, p_W = param[0], param[1]
    except:
        p_H, p_W = param, param

    return p_H, p_W


def median_pool_2d(input, kernel_size, stride, padding, dilation):
    # Input should be 4D (BCHW)
    assert (input.dim() == 4)

    # Get input dimensions
    b_size, c_size, h_size, w_size = input.size()

    # Get input parameters
    k_H, k_W = unpack_param_2d(kernel_size)
    s_H, s_W = unpack_param_2d(stride)
    p_H, p_W = unpack_param_2d(padding)
    d_H, d_W = unpack_param_2d(dilation)

    # First we unfold all the (kernel_size x kernel_size)  patches
    unf_input = F.unfold(input, kernel_size, dilation, padding, stride)

    # Reshape it so that each patch is a column
    row_unf_input = unf_input.reshape(b_size, c_size, k_H * k_W, -1)

    # Apply median operation along the columns for each channel separately
    med_unf_input, med_unf_indexes = torch.median(row_unf_input, dim=2, keepdim=True)

    # Restore original shape
    out_W = math.floor(((w_size + (2 * p_W) - (d_W * (k_W - 1)) - 1) / s_W) + 1)
    out_H = math.floor(((h_size + (2 * p_H) - (d_H * (k_H - 1)) - 1) / s_H) + 1)

    return med_unf_input.reshape(b_size, c_size, out_H, out_W)


class SNRNetwork(nn.Module):
    def __init__(self, in_channels=3, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                 alpha_out=0.5, stride=1, padding=0, dilation=1, groups=1, bias=False, hide_layers=8):
        super(SNRNetwork, self).__init__()
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
                        padding, dilation, groups, bias, activation_layer=nn.PReLU) for i in range(hide_layers * 2))

        self.rescons = nn.ModuleList(ResBlock(hide_channels, hide_channels, kernel_size, alpha_in,
                                              alpha_out, stride, padding, dilation, groups) for i in
                                     range(hide_layers * 2))
        # for input
        self.proin = Conv_BN_ACT(in_channels, hide_channels, kernel_size, 0, alpha_out, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.PReLU)
        # for output
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.proutl = nn.Conv2d(int(alpha_in * hide_channels), out_channels,
                                kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.actl = nn.PReLU()
        self.prouth = nn.Conv2d(hide_channels - int(alpha_in * hide_channels), out_channels,
                                kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.acth = nn.PReLU()
        self.proutc = nn.Conv2d(int(2 * out_channels), out_channels,
                                kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.actc = nn.PReLU()

    def forward(self, x):

        x_h, x_l = self.proin(x)

        for i in range(self.hide_layers * 2):
            x_h, x_l = self.octavecons[i]((x_h, x_l))

#             if i < self.hide_layers:
#                 x_h = median_pool_2d(x_h, self.kernel_size, self.stride, self.padding, self.dilation)
#                 x_l = median_pool_2d(x_l, self.kernel_size, self.stride, self.padding, self.dilation)

            x_h, x_l = self.rescons[i]((x_h, x_l))

        x_l = self.upsample(x_l)
        x_h = self.prouth(x_h)
        x_h = self.act(x_h)
        x_l = self.proutl(x_l)
        x_l = self.act(x_h)
        output = torch.cat((x_h, x_l), 1)
        output = self.proutc(output)
        output = self.act(output)

        return output
