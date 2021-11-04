import torch.nn as nn
import torch.nn.functional as F
import octconv as oct
import venconv as ven
from model.median_pooling import median_pool_2d


class PrePocess(nn.Module):
    def __init__(self, in_channels=1, hide_channels=64, kernel_size=3, alpha=0.5,
                 padding=0, dilation=1, groups=1, bias=False):
        super(PrePocess, self).__init__()
        # parameters
        low_channel_num = [in_channels, 16, 32, int(alpha * hide_channels)]
        high_channel_num = [in_channels, 16, 32, hide_channels - int(alpha * hide_channels)]
        self.pre_num = len(low_channel_num) - 1
        self.input_layers = 2

        # channel expansion
        self.downs = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.prelow = nn.ModuleList(
            ven.Conv_BN_ACT(low_channel_num[i], low_channel_num[i + 1],
                            kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.PReLU)
            for i in range(self.pre_num)
        )
        self.prehigh = nn.ModuleList(
            ven.Conv_BN_ACT(high_channel_num[i], high_channel_num[i + 1],
                            kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.PReLU)
            for i in range(self.pre_num)
        )

        # Pre process layers
        self.inLowLists = nn.ModuleList(
            ven.Conv_BN_ACT(low_channel_num[-1], low_channel_num[-1],
                            kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.PReLU)
            for i in range(self.input_layers))

        self.inHighLists = nn.ModuleList(
            ven.Conv_BN_ACT(high_channel_num[-1], high_channel_num[-1],
                            kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.PReLU)
            for i in range(self.input_layers))

    def forward(self, x):
        # Split input image
        x_l = self.downs(x)
        x_h = x
        # Channel expansion
        for i in range(self.pre_num):
            x_l = self.prelow[i](x_l)
            x_h = self.prehigh[i](x_h)

        # Pre process
        for i in range(self.input_layers):
            x_l = self.inLowLists[i](x_l)
            x_h = self.inHighLists[i](x_h)

        return x_h, x_l


class FeatureExtract(nn.Module):
    def __init__(self, hide_channels=64, kernel_size=3, alpha=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, hide_layers=8):
        super(FeatureExtract, self).__init__()
        # parameters
        self.hide_layers = hide_layers
        high_channel_num = hide_channels - int(alpha * hide_channels)

        # convolution layers
        self.octavecons = nn.ModuleList(oct.Conv_BN_ACT(hide_channels, hide_channels, kernel_size, alpha, alpha, stride,
                            padding, dilation, groups, bias, activation_layer=nn.ReLU) for i in range(hide_layers))
        self.vencons = nn.ModuleList(ven.Conv_BN_ACT(high_channel_num, high_channel_num,
                            kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.PReLU) for i in range(hide_layers))
        # residual layers
        self.rescons = nn.ModuleList(oct.ResBlock(hide_channels, hide_channels, kernel_size, alpha,
                                    alpha, stride, padding, dilation, groups) for i in range(hide_layers))

    def forward(self, x):
        x_h, x_l = x
        for i in range(self.hide_layers):
            x_h, x_l = self.octavecons[i]((x_h, x_l))
            x_h = median_pool_2d(x_h, self.kernel_size, self.stride, self.padding, self.dilation)
            x_h = self.vencons[i](x_h)
            x_h, x_l = self.rescons[i]((x_h, x_l))

        return x_h, x_l


class HighFeature(nn.Module):
    def __init__(self, hide_channels=64, kernel_size=3, alpha=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, hide_layers=8):
        super(HighFeature, self).__init__()
        # parameters
        self.hide_layers = hide_layers
        low_channel_num = int(alpha * hide_channels)
        high_channel_num = hide_channels - int(alpha * hide_channels)

        # convolution layers
        self.octavecons = nn.ModuleList(oct.OctaveHighFeature(hide_channels, hide_channels, kernel_size, alpha, stride,
                            padding, dilation, groups, bias, activation_layer=nn.ReLU) for i in range(hide_layers))
        # channel expansion
        self.lowExpansion = ven.Conv_BN_ACT(low_channel_num, hide_channels,
                            kernel_size, 1, padding, dilation, groups, bias)
        self.highExpansion = ven.Conv_BN_ACT(high_channel_num, hide_channels,
                            kernel_size, 1, padding, dilation, groups, bias)


    def forward(self, x):
        x_h, x_l = x
        x_l = self.lowExpansion(x_l)
        x_h = self.highExpansion(x_h)

        for i in range(self.hide_layers):
            x_h, x_l = self.octavecons[i]((x_h, x_l))

        return x_h, x_l



class Model(nn.Module):
    def __init__(self, in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha=0.5,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Model, self).__init__()
        # parameters
        self.output_layers = 2
        self.preInput = PrePocess(in_channels, hide_channels, kernel_size, alpha, padding, dilation, groups, bias)
        self.featureE = FeatureExtract(hide_channels, kernel_size, alpha, stride, padding, dilation, groups, bias, 4)
        self.highFeature = HighFeature(hide_channels, kernel_size, alpha, stride, padding, dilation, groups, bias, 4)

        # for output
        self.proutl = ven.Conv_BN_ACT(hide_channels, hide_channels,
                                      kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.ReLU
                                      )
        self.prouth = ven.Conv_BN_ACT(hide_channels, hide_channels,
                                      kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.ReLU
                                      )

        self.proutc = ven.Conv_BN_ACT(hide_channels, out_channels,
                                      kernel_size, 1, padding, dilation, groups, bias, activation_layer=nn.ReLU
                                      )

    def forward(self, x):
        x = self.preInput(x)
        x = self.featureE(x)
        x_h, x_l = self.highFeature(x)

        x_h = self.prouth(x_h)

        output = self.proutc(x_h)

        x_l = self.proutl(x_l)

        return output, x_l
