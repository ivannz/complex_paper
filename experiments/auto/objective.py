import torch

from itertools import chain
from torch.nn.modules.loss import _Loss
from cplxmodule.relevance.base import BaseARD


def named_ard_modules(module, prefix=""):
    for name, mod in module.named_modules(prefix=prefix):
        if isinstance(mod, BaseARD):
            yield name, mod


class Penalty(_Loss):
    def __init__(self, mean):
        super().__init__()
        self.reduction = torch.mean if mean else torch.sum

    def forward(self, module, input=None, target=None):
        raise NotImplementedError


class ARDPenalty(Penalty):
    def __init__(self, coef=1., mean=False):
        super().__init__(mean=mean)
        if isinstance(coef, float):
            self.coef = lambda n: coef

        elif isinstance(coef, dict):
            self.coef = coef.get

        elif callable(coef):
            self.coef = coef

    def forward(self, module, input=None, target=None):
        """Reimplements `named_penalties` with non-uniform coefficients."""
        # get names of variational modules and pre-fetch coefficients
        submods = dict(named_ard_modules(module))
        # names, submods = zip(*named_ard_modules(module))  # can't handle empty iterators
        weights = (weight if weight is not None else 1.
                   for weight in map(self.coef, submods.keys()))

        # lazy-compute the weighted sum
        return sum(weight * self.reduction(mod.penalty)
                   for mod, weight in zip(submods.values(), weights)
                   if weight > 0)
