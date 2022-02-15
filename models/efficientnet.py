from torch import nn


class Conv3x3(nn.Module):
    def __init__(self, ch_in, ch_out, k=1, s=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = nn.SiLU()

    def forwark(self, x):
        return self.act(self.bn(self.conv(x)))


class Fused_MBConv(nn.Module):
    def __init__(self):
        pass


class MBConv(nn.Module):
    def __int__(self):
        pass