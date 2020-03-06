import torch
from collections import OrderedDict

from cplxmodule.nn import CplxToCplx
from cplxmodule.nn import CplxConv2d, CplxLinear

from cplxmodule.nn import CplxReal, AsTypeCplx
from cplxmodule.nn.modules.casting import ConcatenatedRealToCplx
from cplxmodule.nn.modules.casting import CplxToConcatenatedReal

# var-dropout
# from cplxmodule.nn.relevance import CplxLinearVD
from cplxmodule.nn.relevance.extensions import CplxLinearVDBogus as CplxLinearVD
# from cplxmodule.nn.relevance import CplxConv2dVD
from cplxmodule.nn.relevance.extensions import CplxConv2dVDBogus as CplxConv2dVD

# automatic relevance determination
from cplxmodule.nn.relevance import CplxLinearARD
from cplxmodule.nn.relevance import CplxConv2dARD

from cplxmodule.nn.masked import CplxConv2dMasked, CplxLinearMasked

from ...musicnet.models.real.base import Flatten


class SimpleConvModel(object):
    Linear = CplxLinear
    Conv2d = CplxConv2d

    def __new__(cls, n_outputs=10, n_inputs=1, upcast=False, half=False):
        if upcast:
            layers = [("cplx", AsTypeCplx())]
        else:
            layers = [("cplx", ConcatenatedRealToCplx(copy=False, dim=-3))]

        n_features = [10, 25, 250] if half else [20, 50, 500]
        layers.extend([
            ("conv1", cls.Conv2d(n_inputs, n_features[0], 5, 1)),
            ("relu1", CplxToCplx[torch.nn.ReLU]()),
            ("pool1", CplxToCplx[torch.nn.AvgPool2d](2, 2)),
            ("conv2", cls.Conv2d(n_features[0], n_features[1], 5, 1)),
            ("relu2", CplxToCplx[torch.nn.ReLU]()),
            ("pool2", CplxToCplx[torch.nn.AvgPool2d](2, 2)),
            ("flat_", CplxToCplx[Flatten](-3, -1)),
            ("lin_1", cls.Linear(4 * 4 * n_features[1], n_features[2])),
            ("relu3", CplxToCplx[torch.nn.ReLU]()),
            ("lin_2", cls.Linear(n_features[2], n_outputs)),
            ("real", CplxReal()),
            # ("real", CplxToConcatenatedReal(dim=-1)),
            # ("lin_3", torch.nn.Linear(20, 10)),
        ])
        return torch.nn.Sequential(OrderedDict(layers))


class SimpleConvModelVD(SimpleConvModel):
    Linear = CplxLinearVD
    Conv2d = CplxConv2dVD


class SimpleConvModelARD(SimpleConvModel):
    Linear = CplxLinearARD
    Conv2d = CplxConv2dARD


class SimpleConvModelMasked(SimpleConvModel):
    Linear = CplxLinearMasked
    Conv2d = CplxConv2dMasked


class SimpleDenseModel(object):
    Linear = CplxLinear

    def __new__(cls, n_outputs=10, n_inputs=1, upcast=False, half=False):
        if upcast:
            layers = [("cplx", AsTypeCplx())]
        else:
            layers = [("cplx", ConcatenatedRealToCplx(copy=False, dim=-3))]

        n_features = [256, 256] if half else [512, 512]
        layers.extend([
            ("flat_", CplxToCplx[Flatten](-3, -1)),
            ("lin_1", cls.Linear(n_inputs * 28 * 28, n_features[0])),
            ("relu2", CplxToCplx[torch.nn.ReLU]()),
            ("lin_2", cls.Linear(n_features[0], n_features[1])),
            ("relu3", CplxToCplx[torch.nn.ReLU]()),
            ("lin_3", cls.Linear(n_features[1], n_outputs)),

            ("real", CplxReal()),
        ])
        return torch.nn.Sequential(OrderedDict(layers))


class SimpleDenseModelVD(SimpleDenseModel):
    Linear = CplxLinearVD


class SimpleDenseModelARD(SimpleDenseModel):
    Linear = CplxLinearARD


class SimpleDenseModelMasked(SimpleDenseModel):
    Linear = CplxLinearMasked


class TwoLayerDenseModel(object):
    Linear = CplxLinear

    def __new__(cls, n_outputs=10, n_inputs=1, upcast=False, half=False):
        if upcast:
            layers = [("cplx", AsTypeCplx())]
        else:
            layers = [("cplx", ConcatenatedRealToCplx(copy=False, dim=-3))]

        n_features = 2048 if half else 4096
        layers.extend([
            ("flat_", CplxToCplx[Flatten](-3, -1)),
            ("lin_1", cls.Linear(n_inputs * 28 * 28, n_features)),
            ("relu2", CplxToCplx[torch.nn.ReLU]()),
            ("lin_2", cls.Linear(n_features, n_outputs)),

            ("real", CplxReal()),
        ])
        return torch.nn.Sequential(OrderedDict(layers))


class TwoLayerDenseModelVD(TwoLayerDenseModel):
    Linear = CplxLinearVD


class TwoLayerDenseModelARD(TwoLayerDenseModel):
    Linear = CplxLinearARD


class TwoLayerDenseModelMasked(TwoLayerDenseModel):
    Linear = CplxLinearMasked
