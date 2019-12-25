from .base import TwoLayerDense
from .base import ShallowConvNet
from .base import DeepConvNet

from cplxmodule.nn.relevance import LinearARD
from cplxmodule.nn.masked import LinearMasked
from cplxmodule.nn.relevance import Conv1dARD
from cplxmodule.nn.masked import Conv1dMasked


class TwoLayerDenseARD(TwoLayerDense):
    Linear = LinearARD


class TwoLayerDenseMasked(TwoLayerDense):
    Linear = LinearMasked


class ShallowConvNetARD(ShallowConvNet):
    Linear = LinearARD
    Conv1d = Conv1dARD


class ShallowConvNetMasked(ShallowConvNet):
    Linear = LinearMasked
    Conv1d = Conv1dMasked


class DeepConvNetARD(DeepConvNet):
    Linear = LinearARD
    Conv1d = Conv1dARD


class DeepConvNetMasked(DeepConvNet):
    Linear = LinearMasked
    Conv1d = Conv1dMasked
