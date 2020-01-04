import torch
from collections import OrderedDict

from cplxmodule.nn import CplxToCplx
from cplxmodule.nn import CplxConv2d, CplxLinear

from cplxmodule.nn.layers import CplxReal, AsTypeCplx
from cplxmodule.nn.layers import ConcatenatedRealToCplx
from cplxmodule.nn.layers import CplxToConcatenatedReal

# from cplxmodule.nn.relevance import CplxConv2dARD, CplxLinearARD
from cplxmodule.nn.relevance.extensions import CplxLinearVDBogus
from cplxmodule.nn.relevance.extensions import CplxConv2dVDBogus

from cplxmodule.nn.masked import CplxConv2dMasked, CplxLinearMasked

from ...musicnet.trabelsi2017.base import Flatten


class SimpleConvModel(object):
    Linear = CplxLinear
    Conv2d = CplxConv2d

    def __new__(cls, n_outputs=10, n_inputs=1, upcast=False):
        if upcast:
            layers = [("cplx", AsTypeCplx())]
        else:
            layers = [("cplx", ConcatenatedRealToCplx(copy=False, dim=-3))]

        layers.extend([
            ("conv1", cls.Conv2d(n_inputs, 20, 5, 1)),
            ("relu1", CplxToCplx[torch.nn.ReLU]()),
            ("pool1", CplxToCplx[torch.nn.AvgPool2d](2, 2)),
            ("conv2", cls.Conv2d(20, 50, 5, 1)),
            ("relu2", CplxToCplx[torch.nn.ReLU]()),
            ("pool2", CplxToCplx[torch.nn.AvgPool2d](2, 2)),
            ("flat_", CplxToCplx[Flatten](-3, -1)),
            ("lin_1", cls.Linear(4 * 4 * 50, 500)),
            ("relu3", CplxToCplx[torch.nn.ReLU]()),
            ("lin_2", cls.Linear(500, n_outputs)),
            ("real", CplxReal()),
            # ("real", CplxToConcatenatedReal(dim=-1)),
            # ("lin_3", torch.nn.Linear(20, 10)),
        ])
        return torch.nn.Sequential(OrderedDict(layers))


class SimpleConvModelARD(SimpleConvModel):
    Linear = CplxLinearVDBogus
    Conv2d = CplxConv2dVDBogus


class SimpleConvModelMasked(SimpleConvModel):
    Linear = CplxLinearMasked
    Conv2d = CplxConv2dMasked


class SimpleDenseModel(object):
    Linear = CplxLinear

    def __new__(cls, n_outputs=10, n_inputs=1, upcast=False):
        if upcast:
            layers = [("cplx", AsTypeCplx())]
        else:
            layers = [("cplx", ConcatenatedRealToCplx(copy=False, dim=-3))]

        layers.extend([
            ("flat_", CplxToCplx[Flatten](-3, -1)),
            ("lin_1", cls.Linear(n_inputs * 28 * 28, 512)),
            ("relu2", CplxToCplx[torch.nn.ReLU]()),
            ("lin_2", cls.Linear(512, 512)),
            ("relu3", CplxToCplx[torch.nn.ReLU]()),
            ("lin_3", cls.Linear(512, n_outputs)),

            ("real", CplxReal()),
        ])
        return torch.nn.Sequential(OrderedDict(layers))


class SimpleDenseModelARD(SimpleDenseModel):
    Linear = CplxLinearVDBogus


class SimpleDenseModelMasked(SimpleDenseModel):
    Linear = CplxLinearMasked


class TwoLayerDenseModel(object):
    Linear = CplxLinear

    def __new__(cls, n_outputs=10, n_inputs=1, upcast=False):
        if upcast:
            layers = [("cplx", AsTypeCplx())]
        else:
            layers = [("cplx", ConcatenatedRealToCplx(copy=False, dim=-3))]

        layers.extend([
            ("flat_", CplxToCplx[Flatten](-3, -1)),
            ("lin_1", cls.Linear(n_inputs * 28 * 28, 4096)),
            ("relu2", CplxToCplx[torch.nn.ReLU]()),
            ("lin_2", cls.Linear(4096, n_outputs)),

            ("real", CplxReal()),
        ])
        return torch.nn.Sequential(OrderedDict(layers))


class TwoLayerDenseModelARD(TwoLayerDenseModel):
    Linear = CplxLinearVDBogus


class TwoLayerDenseModelMasked(TwoLayerDenseModel):
    Linear = CplxLinearMasked
