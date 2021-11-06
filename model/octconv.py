import torch.nn as nn
import math
import torch


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = x_l2h + x_h2h
                x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
                return x_h, x_l
        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        self.act = activation_layer()

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l


class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, alpha_in=0.5,
                 alpha_out=0.5, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ResBlock, self).__init__()

        self.conv1 = Conv_BN_ACT(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)
        self.conv2 = Conv_BN_ACT(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)

    def forward(self, x):
        x_hi, x_li = x

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        x_h = x_h + x_hi
        x_l = x_l + x_li

        return x_h, x_l

class OctaveHighFeature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveHighFeature, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.PixelShuffle(2)
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_channels
        assert 0 <= alpha <= 1, "Alphas should be in the interval from 0 to 1."

        self.conv_l2l = nn.Conv2d(in_channels, int(alpha * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(alpha * groups), bias)
        self.conv_l2h = nn.Conv2d(in_channels, 4 * (out_channels - int(alpha * out_channels)),
                                  kernel_size, 1, padding, dilation, groups, bias)

        self.conv_h2l = nn.Conv2d(in_channels, out_channels - int(alpha * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = nn.Conv2d(in_channels, int(alpha * out_channels),
                                  kernel_size, 1, padding, dilation, math.ceil(groups - alpha * groups), bias)

    def forward(self, x):
        x_h, x_l = x

        x_h = self.downsample(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h)
        x_h2l = self.conv_h2l(self.downsample(x_h))
        if x_l is not None:
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l)
            if self.is_dw:
                return x_h2h, x_l2l
            else:
                x_l2h = self.conv_l2h(x_l)
                x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
                x_h = torch.cat((x_l2h, x_h2h), 1)
                x_l = torch.cat((x_l2l, x_h2l), 1)
                return x_h, x_l
        else:
            return x_h2h, x_h2l


class OctaveHighFeature_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(OctaveHighFeature_BN, self).__init__()
        self.conv = OctaveHighFeature(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                               groups, bias)
        self.bn_h = norm_layer(out_channels)
        self.bn_l = norm_layer(out_channels)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class OctaveHighFeature_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(OctaveHighFeature_BN_ACT, self).__init__()
        self.conv = OctaveHighFeature_BN(in_channels, out_channels, kernel_size, alpha, stride, padding, dilation,
                 groups, bias, norm_layer)
        self.act = activation_layer()

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(x_h)
        x_l = self.act(x_l)
        return x_h, x_l


class ResBlockHighFeature(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
                 alpha=0.5, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(ResBlockHighFeature, self).__init__()

        self.conv1 = OctaveHighFeature_BN_ACT(in_channels, out_channels, kernel_size, alpha, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)
        self.conv2 = OctaveHighFeature_BN_ACT(in_channels, out_channels, kernel_size, alpha, stride,
                                 padding, dilation, groups, bias, activation_layer=nn.ReLU)

    def forward(self, x):
        x_hi, x_li = x

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))

        x_h = x_h + x_hi
        x_l = x_l + x_li

        return x_h, x_l