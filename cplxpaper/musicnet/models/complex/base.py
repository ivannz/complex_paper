import torch
from collections import OrderedDict

from cplxmodule.nn.layers import CplxToCplx, CplxReal, CplxImag
from cplxmodule.nn.layers import ConcatenatedRealToCplx

from cplxmodule.nn.layers import CplxLinear
from cplxmodule.nn.conv import CplxConv1d

from cplxmodule.nn.batchnorm import CplxBatchNorm1d

from ..real.base import Flatten, BodyMixin


class TwoLayerDense(torch.nn.Module):
    """
    n_channels : int
        number of cplx channels (2 x floats).
    """
    Linear = CplxLinear

    def __new__(cls, n_seq=4096, n_channels=1, n_outputs=84):
        return torch.nn.Sequential(OrderedDict([
            # B x C x L float -> B x C/2 x L cplx
            ("cplx", ConcatenatedRealToCplx(copy=False, dim=-2)),
            ("fltn", CplxToCplx[Flatten](1, -1)),
            ("lin1", cls.Linear(n_seq * n_channels, 2048)),
            ("relu", CplxToCplx[torch.nn.ReLU]()),
            ("lin2", cls.Linear(2048, n_outputs)),
            ("real", CplxReal()),
        ]))


def one_conv(n_seq, cls, in_channels, out_channels, kernel, stride, bias=True):
    n_seq = (n_seq - kernel + 1 + stride - 1) // stride  # assumes no padding
    return n_seq, cls(in_channels, out_channels, kernel, stride, bias=bias)


def one_pool(n_seq, cls, kernel, stride):
    n_seq = (n_seq - kernel + 1 + stride - 1) // stride
    return n_seq, CplxToCplx[cls](kernel, stride)


class ShallowConvNet(torch.nn.Module):
    Linear = CplxLinear
    Conv1d = CplxConv1d

    def __new__(cls, n_seq=4096, n_channels=1, n_outputs=84):
        n_seq, conv = one_conv(
            n_seq, cls.Conv1d, n_channels, 32, 512, 16, bias=True)
        n_seq, pool = one_pool(
            n_seq, torch.nn.AvgPool1d, 4, 2)

        return torch.nn.Sequential(OrderedDict([
            ("cplx", ConcatenatedRealToCplx(copy=False, dim=-2)),
            ("conv", conv),
            ("relu", CplxToCplx[torch.nn.ReLU]()),
            ("pool", pool),
            ("fltn", CplxToCplx[Flatten](-2)),
            ("lin1", cls.Linear(n_seq * 32, 2048)),
            ("relu", CplxToCplx[torch.nn.ReLU]()),
            ("lin2", cls.Linear(2048, n_outputs)),
            ("real", CplxReal()),
        ]))


class DeepConvNet(torch.nn.Module):
    Linear = CplxLinear
    Conv1d = CplxConv1d

    @classmethod
    def one_block(cls, n_seq, in_channels, out_channels, kernel, stride, full):
        layers = []
        n_seq, conv = one_conv(
            n_seq, cls.Conv1d, in_channels, out_channels,
            kernel, stride, bias=not full)
        layers.append(("conv", conv))

        if full:
            layers.append(("btch", CplxBatchNorm1d(out_channels)))

        layers.append(("relu", CplxToCplx[torch.nn.ReLU]()))
        if full:
            n_seq, pool = one_pool(n_seq, torch.nn.AvgPool1d, 2, 2)
            layers.append(("pool", pool))

        return n_seq, torch.nn.Sequential(OrderedDict(layers))

    def __new__(cls, n_seq=4096, n_channels=1, n_outputs=84, legacy=True):
        k = 3 if legacy else 6  # sigh, a mistake... had to add `legacy` flag

        param = [
            (  1,  16, k, 2,  True),  # Trabelsi et al. (2017) has kernel=6 (L57)
            ( 16,  32, 3, 2,  True),
            ( 32,  64, 3, 1,  True),
            ( 64,  64, 3, 1,  True),
            ( 64, 128, 3, 1, False),
            (128, 128, 3, 1,  True),
        ]

        named_blocks = []
        for j, par in enumerate(param):
            n_seq, blk = cls.one_block(n_seq, *par)
            named_blocks.append((f"bk{j:02d}", blk))

        return torch.nn.Sequential(OrderedDict([
            ("cplx", ConcatenatedRealToCplx(copy=False, dim=-2)),
            *named_blocks,
            ("fltn", CplxToCplx[Flatten](-2)),
            ("lin1", cls.Linear(n_seq * 128, 2048)),
            ("relu", CplxToCplx[torch.nn.ReLU]()),
            ("lin2", cls.Linear(2048, n_outputs)),
            ("real", CplxReal()),
        ]))
