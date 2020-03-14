r"""The real valued VGG model. Adapted from 

https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
https://github.com/anokland/local-loss/blob/master/train.py
"""
import torch

# var-dropout
from cplxmodule.nn.relevance import LinearVD
from cplxmodule.nn.relevance import Conv2dVD

# automatic relevance determination
from cplxmodule.nn.relevance import LinearARD
from cplxmodule.nn.relevance import Conv2dARD

from cplxmodule.nn.masked import Conv2dMasked, LinearMasked

from ....musicnet.models.real.base import Flatten


cfg = {
# 'VGG6a'  : [ 128,                'M', 256,      'M', 512,                'M', 512                                             ],
# 'VGG6b'  : [ 128,                'M', 256,      'M', 512,                'M', 512,                'M'                         ],
'VGG8'   : [  64,                'M', 128,      'M', 256,                'M', 512,                'M', 512,                'M'],
# 'VGG8a'  : [ 128, 256,           'M', 256, 512, 'M', 512,                'M', 512                                             ],
# 'VGG8b'  : [ 128, 256,           'M', 256, 512, 'M', 512,                'M', 512,                'M'                         ],
# 'VGG11b' : [ 128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512,           'M', 512,                'M'                         ],
'VGG11'  : [  64,                'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
'VGG13'  : [  64,  64,           'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
'VGG16'  : [  64,  64,           'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
'VGG19'  : [  64,  64,           'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(torch.nn.Module):
    Linear = torch.nn.Linear
    Conv2d = torch.nn.Conv2d

    def __new__(cls, vgg_name='VGG16', n_outputs=10, n_channels=3, double=False):
        layers = []
        for x in cfg[vgg_name]:
            if x == 'M':
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

            else:
                x = (x * 2) if double else x
                layers.extend([
                    cls.Conv2d(n_channels, x, kernel_size=3, padding=1),
                    torch.nn.BatchNorm2d(x),
                    torch.nn.ReLU(),
                ])
                n_channels = x

        return torch.nn.Sequential(
            *layers,
            Flatten(-3, -1),
            cls.Linear(1024 if double else 512, n_outputs)
        )


class VGGVD(VGG):
    Linear = LinearVD
    Conv2d = Conv2dVD


class VGGARD(VGG):
    Linear = LinearARD
    Conv2d = Conv2dARD


class VGGMasked(VGG):
    Linear = LinearMasked
    Conv2d = Conv2dMasked
