import torch
from collections import OrderedDict

from cplxmodule.nn import CplxToCplx
from cplxmodule.nn import CplxConv2d, CplxLinear

from cplxmodule.nn.layers import CplxReal
from cplxmodule.nn.layers import ConcatenatedRealToCplx
from cplxmodule.nn.layers import CplxToConcatenatedReal
from cplxmodule.nn.relevance import CplxConv2dARD, CplxLinearARD
from cplxmodule.nn.masked import CplxConv2dMasked, CplxLinearMasked


class MNISTModel(torch.nn.Sequential):
    Linear = CplxLinear
    Conv2d = CplxConv2d

    def __init__(self):
        layers = [
            ("cplx", ConcatenatedRealToCplx(copy=False, dim=-3)),

            ("conv1", self.Conv2d( 1, 20, 5, 1)),
            ("relu1", CplxToCplx[torch.nn.ReLU]()),
            ("pool1", CplxToCplx[torch.nn.AvgPool2d](2, 2)),
            ("conv2", self.Conv2d(20, 50, 5, 1)),
            ("relu2", CplxToCplx[torch.nn.ReLU]()),
            ("pool2", CplxToCplx[torch.nn.AvgPool2d](2, 2)),
            ("flat_", CplxToCplx[torch.nn.Flatten](-3, -1)),
            ("lin_1", self.Linear(4 * 4 * 50, 500)),
            ("relu3", CplxToCplx[torch.nn.ReLU]()),
            ("lin_2", self.Linear(500, 10)),
            ("real", CplxReal()),
            # ("real", CplxToConcatenatedReal(dim=-1)),
            # ("lin_3", torch.nn.Linear(20, 10)),
        ]

        super().__init__(OrderedDict(layers))


class MNISTModelARD(MNISTModel):
    Linear = CplxLinearARD
    Conv2d = CplxConv2dARD


class MNISTModelMasked(MNISTModel):
    Linear = CplxLinearMasked
    Conv2d = CplxConv2dMasked
