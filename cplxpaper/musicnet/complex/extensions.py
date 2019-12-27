from .base import CplxTwoLayerDense
from .base import CplxShallowConvNet
from .base import CplxDeepConvNet

from cplxmodule.nn.relevance.extensions import CplxLinearVDBogus
from cplxmodule.nn.masked import CplxLinearMasked
from cplxmodule.nn.relevance.extensions import CplxConv1dVDBogus
from cplxmodule.nn.masked import CplxConv1dMasked


class CplxTwoLayerDenseARD(CplxTwoLayerDense):
    Linear = CplxLinearVDBogus


class CplxTwoLayerDenseMasked(CplxTwoLayerDense):
    Linear = CplxLinearMasked


class CplxShallowConvNetARD(CplxShallowConvNet):
    Linear = CplxLinearVDBogus
    Conv1d = CplxConv1dVDBogus


class CplxShallowConvNetMasked(CplxShallowConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked


class CplxDeepConvNetARD(CplxDeepConvNet):
    Linear = CplxLinearVDBogus
    Conv1d = CplxConv1dVDBogus


class CplxDeepConvNetMasked(CplxDeepConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked
