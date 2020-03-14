# var-dropout
from cplxmodule.nn.relevance import LinearVD
from cplxmodule.nn.relevance import Conv1dVD

# automatic relevance determination
from cplxmodule.nn.relevance import LinearARD
from cplxmodule.nn.relevance import Conv1dARD

from cplxmodule.nn.masked import LinearMasked
from cplxmodule.nn.masked import Conv1dMasked

from .base import TwoLayerDense
from .base import ShallowConvNet
from .base import DeepConvNet


class TwoLayerDenseVD(TwoLayerDense):
    Linear = LinearVD


class TwoLayerDenseARD(TwoLayerDense):
    Linear = LinearARD


class TwoLayerDenseMasked(TwoLayerDense):
    Linear = LinearMasked


class ShallowConvNetVD(ShallowConvNet):
    Linear = LinearVD
    Conv1d = Conv1dVD


class ShallowConvNetARD(ShallowConvNet):
    Linear = LinearARD
    Conv1d = Conv1dARD


class ShallowConvNetMasked(ShallowConvNet):
    Linear = LinearMasked
    Conv1d = Conv1dMasked


class DeepConvNetVD(DeepConvNet):
    Linear = LinearVD
    Conv1d = Conv1dVD


class DeepConvNetARD(DeepConvNet):
    Linear = LinearARD
    Conv1d = Conv1dARD


class DeepConvNetMasked(DeepConvNet):
    Linear = LinearMasked
    Conv1d = Conv1dMasked
