

###### please install detron2 using command as bellow:
####     python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


import torch
import torch.nn as nn
from detectron2.layers import ModulatedDeformConv, DeformConv
import torch.nn.functional as F


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError
     
        
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class GSA(nn.Module):
    def __init__(self, n_feats):
        super(GSA, self).__init__()
        activation = 'relu'
        self.F_f = nn.Sequential(
            nn.Linear(2 * n_feats, 6 * n_feats),  ## !!!!!!
            actFunc(activation),
            nn.Linear(6 * n_feats, 2 * n_feats),  ## !!!!!!
            nn.Sigmoid()
        )
        # condense layer
        self.condense = conv1x1(2 * n_feats, n_feats)## !!!!
        self.act = actFunc(activation)

    def forward(self, cor):
        w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()
        if len(w.shape) == 1:
            w = w.unsqueeze(dim=0)
        w = self.F_f(w)
        w = w.reshape(*w.shape, 1, 1)
        out = self.act(self.condense(w * cor))

        return out



class ModulatedDeformLayer(nn.Module):
    """
    Modulated Deformable Convolution (v2)
    """

    def __init__(self, in_chs, out_chs, kernel_size=3, deformable_groups=1, activation='relu'):
        super(ModulatedDeformLayer, self).__init__()
        assert isinstance(kernel_size, (int, list, tuple))
        self.deform_offset = conv3x3(in_chs, (3 * kernel_size ** 2) * deformable_groups)
        self.act = actFunc(activation)
        self.deform = ModulatedDeformConv(
            in_chs,
            out_chs,
            kernel_size,
            stride=1,
            padding=1,
            deformable_groups=deformable_groups
        )

    def forward(self, x, feat):
        offset_mask = self.deform_offset(feat)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        out = self.deform(x, offset, mask)
        out = self.act(out)
        return out


class FusionModule(nn.Module):
    def __init__(self, n_feats):
        super(FusionModule, self).__init__()
        self.attention = GSA(n_feats)
        self.deform = ModulatedDeformLayer(n_feats, n_feats, deformable_groups=2)  # deformable_groups must be a divisor of n_feats

    def forward(self, x):
        x = self.attention(x)
        return self.deform(x, x)


if __name__ == '__main__':

    init_chs = 6
    fusion2 = FusionModule(init_chs).cuda()  ## input_channel 
    
    p_warped2 = torch.randn(2, 6, 256, 256).cuda()
    f_warped2 = torch.randn(2, 6, 256, 256).cuda()
    x2 = torch.cat([p_warped2, f_warped2], dim=1)
    x2 = fusion2(x2)  # (B,init_chs, H,W)
    
    #import ipdb;ipdb.set_trace()
    print('Finished!!!!!')