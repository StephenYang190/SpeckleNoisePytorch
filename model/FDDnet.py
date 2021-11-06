import torch.nn as nn
import model.octconv as oct
import model.venconv as ven
from model.median_pooling import median_pool_2d


class PrePocess(nn.Module):
    def __init__(self, config):
        super(PrePocess, self).__init__()
        # parameters
        low_channel_num = [config["in_channels"], 16, 32, config["lcn"]]
        high_channel_num = [config["in_channels"], 16, 32, config["hcn"]]
        self.pre_num = len(low_channel_num) - 1
        self.input_layers = 2

        # channel expansion
        self.downs = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.prelow = nn.ModuleList(
            ven.Conv_BN_ACT(low_channel_num[i], low_channel_num[i + 1],
                            config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.PReLU)
            for i in range(self.pre_num)
        )
        self.prehigh = nn.ModuleList(
            ven.Conv_BN_ACT(high_channel_num[i], high_channel_num[i + 1],
                            config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.PReLU)
            for i in range(self.pre_num)
        )

        # Pre process layers
        self.inLowLists = nn.ModuleList(
            ven.Conv_BN_ACT(low_channel_num[-1], low_channel_num[-1],
                            config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.PReLU)
            for i in range(self.input_layers))

        self.inHighLists = nn.ModuleList(
            ven.Conv_BN_ACT(high_channel_num[-1], high_channel_num[-1],
                            config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.PReLU)
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
    def __init__(self, config):
        super(FeatureExtract, self).__init__()
        # parameters
        self.hide_layers = config["hide_layers"]
        self.config = config

        # convolution layers
        self.octavecons = nn.ModuleList(oct.Conv_BN_ACT(config["hide_channels"], config["hide_channels"], config["kernel_size"], config["alpha"], config["alpha"], config["stride"],
                            config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.ReLU) for i in range(self.hide_layers))
        self.vencons = nn.ModuleList(ven.Conv_BN_ACT(config["hcn"], config["hcn"],
                            config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.PReLU) for i in range(self.hide_layers))
        # residual layers
        self.rescons = nn.ModuleList(oct.ResBlock(config["hide_channels"], config["hide_channels"], config["kernel_size"], config["alpha"],
                                    config["alpha"], config["stride"], config["padding"], config["dilation"], config["groups"]) for i in range(self.hide_layers))

    def forward(self, x):
        x_h, x_l = x
        for i in range(self.hide_layers):
            x_h, x_l = self.octavecons[i]((x_h, x_l))
            x_h = median_pool_2d(x_h, self.config["kernel_size"], self.config["stride"], self.config["padding"], self.config["dilation"])
            x_h = self.vencons[i](x_h)
            x_h, x_l = self.rescons[i]((x_h, x_l))

        return x_h, x_l


class HighFeature(nn.Module):
    def __init__(self, config):
        super(HighFeature, self).__init__()
        # parameters
        self.hide_layers = config["hide_layers"]

        # convolution layers
        self.octavecons = nn.ModuleList(oct.OctaveHighFeature_BN_ACT(config["hide_channels"], config["hide_channels"], config["kernel_size"], config["alpha"], config["stride"],
                            config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.ReLU) for i in range(self.hide_layers))
        # channel expansion
        self.lowExpansion = ven.Conv_BN_ACT(config["lcn"], config["hide_channels"],
                            config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"])
        self.highExpansion = ven.Conv_BN_ACT(config["hcn"], config["hide_channels"],
                            config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"])


    def forward(self, x):
        x_h, x_l = x
        x_l = self.lowExpansion(x_l)
        x_h = self.highExpansion(x_h)

        for i in range(self.hide_layers):
            x_h, x_l = self.octavecons[i]((x_h, x_l))

        return x_h, x_l


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # parameters
        self.output_layers = 2
        lcn = int(config["alpha"] * config["hide_channels"])
        hcn = config["hide_channels"] - lcn
        config["lcn"] = lcn
        config["hcn"] = hcn
        self.config = config

        # Process
        self.preInput = PrePocess(config)
        self.featureE = FeatureExtract(config)
        self.highFeature = HighFeature(config)
        # for output
        self.proutl = ven.Conv_BN_ACT(config["hide_channels"], config["out_channels"],
                                      config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.ReLU
                                      )
        self.prouth = ven.Conv_BN_ACT(config["hide_channels"], config["hide_channels"],
                                      config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.ReLU
                                      )

        self.proutc = ven.Conv_BN_ACT(config["hide_channels"], config["out_channels"],
                                      config["kernel_size"], 1, config["padding"], config["dilation"], config["groups"], config["bias"], activation_layer=nn.ReLU
                                      )

    def forward(self, x):
        x = self.preInput(x)
        x = self.featureE(x)
        x_h, x_l = self.highFeature(x)

        x_h = self.prouth(x_h)

        output = self.proutc(x_h)

        x_l = self.proutl(x_l)

        return output, x_l
