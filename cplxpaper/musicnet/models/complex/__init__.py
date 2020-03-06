# var-dropout
# from cplxmodule.nn.relevance import CplxLinearVD
from cplxmodule.nn.relevance.extensions import CplxLinearVDBogus as CplxLinearVD
# from cplxmodule.nn.relevance import CplxConv1dVD
from cplxmodule.nn.relevance.extensions import CplxConv1dVDBogus as CplxConv1dVD

# automatic relevance determination
from cplxmodule.nn.relevance import CplxLinearARD
from cplxmodule.nn.relevance import CplxConv1dARD

from cplxmodule.nn.masked import CplxLinearMasked
from cplxmodule.nn.masked import CplxConv1dMasked

from .base import TwoLayerDense
from .base import ShallowConvNet
from .base import DeepConvNet


class TwoLayerDenseVD(TwoLayerDense):
    Linear = CplxLinearVD


class TwoLayerDenseARD(TwoLayerDense):
    Linear = CplxLinearARD


class TwoLayerDenseMasked(TwoLayerDense):
    Linear = CplxLinearMasked


class ShallowConvNetVD(ShallowConvNet):
    Linear = CplxLinearVD
    Conv1d = CplxConv1dVD


class ShallowConvNetARD(ShallowConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class ShallowConvNetMasked(ShallowConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked


class DeepConvNetVD(DeepConvNet):
    Linear = CplxLinearVD
    Conv1d = CplxConv1dVD


class DeepConvNetARD(DeepConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class DeepConvNetMasked(DeepConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked
