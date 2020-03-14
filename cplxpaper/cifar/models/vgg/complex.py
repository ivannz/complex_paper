"""The complex valued VGG model."""
import math
import torch

from collections import OrderedDict

from cplxmodule import cplx
from cplxmodule.nn import CplxToCplx
from cplxmodule.nn import CplxReal, AsTypeCplx
from cplxmodule.nn.modules.casting import ConcatenatedRealToCplx

from cplxmodule.nn import CplxLinear
from cplxmodule.nn import CplxConv2d
from cplxmodule.nn import CplxBatchNorm2d

# var-dropout
# from cplxmodule.nn.relevance import CplxLinearVD
from cplxmodule.nn.relevance.extensions import CplxLinearVDBogus as CplxLinearVD
# from cplxmodule.nn.relevance import CplxConv2dVD
from cplxmodule.nn.relevance.extensions import CplxConv2dVDBogus as CplxConv2dVD

# automatic relevance determination
from cplxmodule.nn.relevance import CplxLinearARD
from cplxmodule.nn.relevance import CplxConv2dARD

from cplxmodule.nn.masked import CplxConv2dMasked, CplxLinearMasked

from ....musicnet.models.real.base import Flatten

from .real import cfg


class VGG(torch.nn.Module):
    Linear = CplxLinear
    Conv2d = CplxConv2d

    def __new__(cls, vgg_name='VGG16', n_outputs=10, n_channels=3,
                upcast=False, half=False):
        if upcast:
            layers = [AsTypeCplx()]
        else:
            layers = [ConcatenatedRealToCplx(copy=False, dim=-3)]

        for x in cfg[vgg_name]:
            if x == 'M':
                layers.append(CplxToCplx[torch.nn.MaxPool2d](kernel_size=2, stride=2))

            else:
                x = (x // 2) if half else x
                layers.extend([
                    cls.Conv2d(n_channels, x, kernel_size=3, padding=1),
                    CplxBatchNorm2d(x),
                    CplxToCplx[torch.nn.ReLU](),
                ])
                n_channels = x

        # the last integer x was 512 (or 256).
        return torch.nn.Sequential(
            *layers,
            CplxToCplx[Flatten](-3, -1),
            cls.Linear(256 if half else 512, n_outputs),
            CplxReal()
        )


class VGGVD(VGG):
    Linear = CplxLinearVD
    Conv2d = CplxConv2dVD


class VGGARD(VGG):
    Linear = CplxLinearARD
    Conv2d = CplxConv2dARD


class VGGMasked(VGG):
    Linear = CplxLinearMasked
    Conv2d = CplxConv2dMasked
