import torch
from collections import OrderedDict

# var-dropout
from cplxmodule.nn.relevance.extensions import LinearVD
from cplxmodule.nn.relevance.extensions import Conv2dVD

# automatic relevance determination
from cplxmodule.nn.relevance.extensions import LinearARD
from cplxmodule.nn.relevance.extensions import Conv2dARD

from cplxmodule.nn.masked import Conv2dMasked, LinearMasked

from ...musicnet.models.real.base import Flatten


class SimpleConvModel(object):
    """Simple convolutional model for MNIST.

    Use `__new__` returning a Sequential to mimic a smart model constructor,
    and simplify object reference via strings in JSON manifests.
    """
    Linear = torch.nn.Linear
    Conv2d = torch.nn.Conv2d

    def __new__(cls, n_outputs=10, n_inputs=1, double=False):
        n_features = [40, 100, 1000] if double else [20, 50, 500]
        return torch.nn.Sequential(OrderedDict([
            ("conv1", cls.Conv2d(n_inputs, n_features[0], 5, 1)),
            ("relu1", torch.nn.ReLU()),
            ("pool1", torch.nn.AvgPool2d(2, 2)),
            ("conv2", cls.Conv2d(n_features[0], n_features[1], 5, 1)),
            ("relu2", torch.nn.ReLU()),
            ("pool2", torch.nn.AvgPool2d(2, 2)),
            ("flat_", Flatten(-3, -1)),
            ("lin_1", cls.Linear(4 * 4 * n_features[1], n_features[2])),
            ("relu3", torch.nn.ReLU()),
            ("lin_2", cls.Linear(n_features[2], n_outputs)),
        ]))


class SimpleConvModelVD(SimpleConvModel):
    Linear = LinearVD
    Conv2d = Conv2dVD


class SimpleConvModelARD(SimpleConvModel):
    Linear = LinearARD
    Conv2d = Conv2dARD


class SimpleConvModelMasked(SimpleConvModel):
    Linear = LinearMasked
    Conv2d = Conv2dMasked


class SimpleDenseModel(object):
    Linear = torch.nn.Linear

    def __new__(cls, n_outputs=10, n_inputs=1, double=False):
        n_features = [1024, 1024] if double else [512, 512]
        return torch.nn.Sequential(OrderedDict([
            ("flat_", Flatten(-3, -1)),
            ("lin_1", cls.Linear(n_inputs * 28 * 28, n_features[0])),
            ("relu2", torch.nn.ReLU()),
            ("lin_2", cls.Linear(n_features[0], n_features[1])),
            ("relu3", torch.nn.ReLU()),
            ("lin_3", cls.Linear(n_features[1], n_outputs)),
        ]))


class SimpleDenseModelVD(SimpleDenseModel):
    Linear = LinearVD


class SimpleDenseModelARD(SimpleDenseModel):
    Linear = LinearARD


class SimpleDenseModelMasked(SimpleDenseModel):
    Linear = LinearMasked


class TwoLayerDenseModel(object):
    Linear = torch.nn.Linear

    def __new__(cls, n_outputs=10, n_inputs=1, double=False):
        n_features = 8192 if double else 4096
        return torch.nn.Sequential(OrderedDict([
            ("flat_", Flatten(-3, -1)),
            ("lin_1", cls.Linear(n_inputs * 28 * 28, n_features)),
            ("relu2", torch.nn.ReLU()),
            ("lin_2", cls.Linear(n_features, n_outputs)),
        ]))


class TwoLayerDenseModelVD(TwoLayerDenseModel):
    Linear = LinearVD


class TwoLayerDenseModelARD(TwoLayerDenseModel):
    Linear = LinearARD


class TwoLayerDenseModelMasked(TwoLayerDenseModel):
    Linear = LinearMasked
