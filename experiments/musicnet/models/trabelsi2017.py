"""Reimplementation of model used by Trabelsi et al. (2017)
"""
import torch

from collections import OrderedDict
from torch.nn import init


def reset_bias(mod):
    if hasattr(mod, "bias"):
        init.constant_(mod.bias, -5)


class Flatten(torch.nn.Module):
    def __init__(self, start_dim=0, end_dim=-1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)


class BodyMixin(torch.nn.Module):
    def forward(self, input):
        # just delegate to self.body
        return self.body(input)


class TwoLayerDense(BodyMixin, torch.nn.Module):
    Linear = torch.nn.Linear

    def __init__(self, n_seq=4096, n_channels=2, n_outputs=84):
        super().__init__()
        # assume zero-th batch dimension
        self.body = torch.nn.Sequential(OrderedDict([
            ("flatten", Flatten(1, -1)),
            ("dense00", self.Linear(n_seq * n_channels, 2048)),
            ("activ01", torch.nn.ReLU()),
            ("dense01", self.Linear(2048, n_outputs)),
            # do not use sigmoids: outputs are fed into BCEwithLogits
        ]))

        reset_bias(self.body[-1])


class ShallowConvNet(BodyMixin, torch.nn.Module):
    Linear = torch.nn.Linear
    Conv1d = torch.nn.Conv1d

    def __init__(self, n_seq=4096, n_channels=2, n_outputs=84):
        super().__init__()
        # keras's dynamic input dependent construction is neat.

        n_seq = (n_seq - 512 + 1 + 16 - 1) // 16  # tailored to `cnv1d00`
        n_seq = (n_seq -   4 + 1 +  2 - 1) //  2  # tailored to `max1d01`
        self.body = torch.nn.Sequential(OrderedDict([
            ("cnv1d00", self.Conv1d(n_channels, 64, 512, stride=16)),
            ("activ01", torch.nn.ReLU()),
            ("max1d01", torch.nn.MaxPool1d(4, stride=2)),
            ("flatten", Flatten(1, -1)),
            ("dense02", self.Linear(n_seq * 64, 2048)),
            ("activ03", torch.nn.ReLU()),
            ("dense03", self.Linear(2048, n_outputs)),
        ]))

        reset_bias(self.body[-1])


class DeepConvNet(BodyMixin, torch.nn.Module):
    Linear = torch.nn.Linear
    Conv1d = torch.nn.Conv1d
    BatchNorm1d = torch.nn.BatchNorm1d

    @classmethod
    def one_block(cls, in_features, out_features, kernel, stride, full=True):
        layers = [("conv", cls.Conv1d(in_features, out_features, kernel, stride))]
        if full:
            layers.append(("btch", cls.BatchNorm1d(out_features)))

        layers.append(("relu", torch.nn.ReLU()))
        if full:
            layers.append(("maxp", torch.nn.MaxPool1d(2, stride=2)))
        return torch.nn.Sequential(OrderedDict(layers))

    def __init__(self, n_seq=4096, n_channels=2, n_outputs=84):
        super().__init__()
        # keras's dynamic input dependent construction is neat.
        n_ker = [7, 3, 3, 3, 3, 3]
        n_str = [3, 2, 1, 1, 1, 1]
        for j, (k, s) in enumerate(zip(n_ker, n_str)):
            n_seq = (n_seq - k + 1 + s - 1) // s
            if j != 4:  # the fourth block has no maxPool
                n_seq = (n_seq - 2 + 1 + 2 - 1) // 2

        self.body = torch.nn.Sequential(OrderedDict([
            ("block00", self.one_block(
                                n_channels,  64, n_ker[0], n_str[0],  True)),
            ("block01", self.one_block( 64,  64, n_ker[1], n_str[1],  True)),
            ("block02", self.one_block( 64, 128, n_ker[2], n_str[2],  True)),
            ("block03", self.one_block(128, 128, n_ker[3], n_str[3],  True)),
            ("block04", self.one_block(128, 256, n_ker[4], n_str[4], False)),
            ("block05", self.one_block(256, 256, n_ker[5], n_str[5],  True)),
            ("flatten", Flatten(1, -1)),
            ("dense06", self.Linear(n_seq * 256, 2048)),
            ("activ07", torch.nn.ReLU()),
            ("dense07", self.Linear(2048, n_outputs)),
        ]))

        reset_bias(self.body[-1])
