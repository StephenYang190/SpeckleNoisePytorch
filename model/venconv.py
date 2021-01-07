import torch
import torch.nn as nn

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels),
                        kernel_size, 1, padding, dilation, groups, bias)
        self.bn = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = Conv_BN(in_channels, out_channels, kernel_size, stride, padding, dilation,
                               groups, bias, norm_layer)
        self.act = activation_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
                stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ResBlock, self).__init__()

        self.conv1 = Conv_BN_ACT(in_channels, out_channels, kernel_size, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)
        self.conv2 = Conv_BN_ACT(in_channels, out_channels, kernel_size, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)

    def forward(self, x):
        x_in = x

        x = self.conv1(x)
        x = self.conv2(x)

        x_out = x + x_in

        return x_out