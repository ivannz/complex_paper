# var-dropout
from cplxmodule.nn.relevance.extensions import CplxLinearVDBogus
from cplxmodule.nn.relevance.extensions import CplxConv1dVDBogus

# automatic relevance determination
from cplxmodule.nn.relevance.extensions import CplxLinearARD
from cplxmodule.nn.relevance.extensions import CplxConv1dARD

from cplxmodule.nn.masked import CplxLinearMasked
from cplxmodule.nn.masked import CplxConv1dMasked

from .base import TwoLayerDense
from .base import ShallowConvNet
from .base import DeepConvNet


class TwoLayerDenseVD(TwoLayerDense):
    Linear = CplxLinearVDBogus


class TwoLayerDenseARD(TwoLayerDense):
    Linear = CplxLinearARD


class TwoLayerDenseMasked(TwoLayerDense):
    Linear = CplxLinearMasked


class ShallowConvNetVD(ShallowConvNet):
    Linear = CplxLinearVDBogus
    Conv1d = CplxConv1dVDBogus


class ShallowConvNetARD(ShallowConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class ShallowConvNetMasked(ShallowConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked


class DeepConvNetVD(DeepConvNet):
    Linear = CplxLinearVDBogus
    Conv1d = CplxConv1dVDBogus


class DeepConvNetARD(DeepConvNet):
    Linear = CplxLinearARD
    Conv1d = CplxConv1dARD


class DeepConvNetMasked(DeepConvNet):
    Linear = CplxLinearMasked
    Conv1d = CplxConv1dMasked
