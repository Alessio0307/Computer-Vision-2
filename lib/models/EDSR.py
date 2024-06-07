import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feats, n_feats, kernel_size, bias=bias, padding=1))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:    # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class EDSR(nn.Module):
    def __init__(self, num_channels=256, out_channels=3, scale=1, n_resblock=16, n_feats=64, kernel_size=3):
        super(EDSR, self).__init__()
        conv = nn.Conv2d

        act = nn.ReLU(True)

        # define head module
        m_head = [conv(num_channels, n_feats, kernel_size, padding=1)]

        # define body module
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=1.) for _ in range(n_resblock)]
        m_body.append(conv(n_feats, n_feats, kernel_size, padding=1))

        # define tail module
        m_tail = [conv(n_feats, out_channels, kernel_size, padding=1)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))