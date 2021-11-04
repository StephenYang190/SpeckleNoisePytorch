def buildModel(op):
    if(op == 'om'):
        from model.SNRom import SNRom
        return SNRom(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                         alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
    elif(op == 'o'):
        from model.SNRo import SNRo
        return SNRo(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha_in=0.5,
                        alpha_out=0.5, stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
    elif(op == 'c'):
        from model.SNRc import SNRc
        return SNRc(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3,
                        stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
    elif(op == 'fdd'):
        from model.FDDnet import Model
        return Model(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3, alpha=0.5,
                         stride=1, padding=1, dilation=1, groups=1, bias=False)
    else:
        from model.SNRcm import SNRcm
        return SNRcm(in_channels=1, hide_channels=64, out_channels=1, kernel_size=3,
                         stride=1, padding=1, dilation=1, groups=1, bias=False, hide_layers=8)
