from .base import CplxTwoLayerDense
from .base import CplxShallowConvNet
from .base import CplxDeepConvNet

from cplxmodule.relevance import CplxLinearARD
from cplxmodule.masked import CplxLinearMasked
from cplxmodule.relevance import CplxConv1dARD
from cplxmodule.masked import CplxConv1dMasked


class CplxTwoLayerDenseARD(CplxTwoLayerDense):
    Linear = CplxLinearARD


class CplxTwoLayerDenseMasked(CplxTwoLayerDense):
    Linear = CplxLinearMasked


class CplxShallowConvNetARD(CplxShallowConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class CplxShallowConvNetMasked(CplxShallowConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked


class CplxDeepConvNetARD(CplxDeepConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class CplxDeepConvNetMasked(CplxDeepConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked
