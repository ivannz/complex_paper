from .base import CplxTwoLayerDense
from .base import CplxShallowConvNet
from .base import CplxDeepConvNet

# var-dropout
from cplxmodule.nn.relevance.extensions import CplxLinearVDBogus
from cplxmodule.nn.relevance.extensions import CplxConv1dVDBogus

# automatic relevance determination
from cplxmodule.nn.relevance.extensions import CplxLinearARD
from cplxmodule.nn.relevance.extensions import CplxConv1dARD

from cplxmodule.nn.masked import CplxLinearMasked
from cplxmodule.nn.masked import CplxConv1dMasked


class CplxTwoLayerDenseVD(CplxTwoLayerDense):
    Linear = CplxLinearVDBogus


class CplxTwoLayerDenseARD(CplxTwoLayerDense):
    Linear = CplxLinearARD


class CplxTwoLayerDenseMasked(CplxTwoLayerDense):
    Linear = CplxLinearMasked


class CplxShallowConvNetVD(CplxShallowConvNet):
    Linear = CplxLinearVDBogus
    Conv1d = CplxConv1dVDBogus


class CplxShallowConvNetARD(CplxShallowConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class CplxShallowConvNetMasked(CplxShallowConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked


class CplxDeepConvNetVD(CplxDeepConvNet):
    Linear = CplxLinearVDBogus
    Conv1d = CplxConv1dVDBogus


class CplxDeepConvNetARD(CplxDeepConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class CplxDeepConvNetMasked(CplxDeepConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked
