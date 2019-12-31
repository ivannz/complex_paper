import torch
from collections import OrderedDict

from cplxmodule.nn.relevance import Conv2dARD, LinearARD
from cplxmodule.nn.masked import Conv2dMasked, LinearMasked


class MNISTModel(torch.nn.Sequential):
    Linear = torch.nn.Linear
    Conv2d = torch.nn.Conv2d

    def __init__(self):
        layers = [
            ("conv1", self.Conv2d( 1, 20, 5, 1)),
            ("relu1", torch.nn.ReLU()),
            ("pool1", torch.nn.AvgPool2d(2, 2)),
            ("conv2", self.Conv2d(20, 50, 5, 1)),
            ("relu2", torch.nn.ReLU()),
            ("pool2", torch.nn.AvgPool2d(2, 2)),
            ("flat_", torch.nn.Flatten(-3, -1)),
            ("lin_1", self.Linear(4 * 4 * 50, 500)),
            ("relu3", torch.nn.ReLU()),
            ("lin_2", self.Linear(500, 10)),
        ]

        super().__init__(OrderedDict(layers))


class MNISTModelARD(MNISTModel):
    Linear = LinearARD
    Conv2d = Conv2dARD


class MNISTModelMasked(MNISTModel):
    Linear = LinearMasked
    Conv2d = Conv2dMasked
