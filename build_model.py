import yaml


def buildModel(op):
    with open("./config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    if (op == 'om'):
        from model.SNRom import SNRom
        return SNRom(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                     alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
    elif (op == 'o'):
        from model.SNRo import SNRo
        return SNRo(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                    alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
    elif (op == 'c'):
        from model.SNRc import SNRc
        return SNRc(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3,
                    stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
    elif (op == 'cm'):
        from model.SNRcm import SNRcm
        return SNRcm(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3,
                     stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
    else:
        from model.FDDnet import Model
        return Model(config)
