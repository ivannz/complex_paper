import os
import copy
import time
import gzip
import tqdm
import math
import json

import torch
import numpy as np

from setuptools._vendor.packaging.version import Version

from .delayed import DelayedKeyboardInterrupt
from .utils import get_instance


__version__ = "0.1s"


def save_snapshot(filename, **kwargs):
    with gzip.open(filename, "wb", compresslevel=5) as fout:
        torch.save(dict(kwargs, **{
            "__version__": __version__,
            "__timestamp__": time.strftime("%Y%m%d-%H%M%S"),
        }), fout, pickle_protocol=3)

    return filename


def get_optimizer(parameters, *, lr, **options):
    return get_instance(parameters, lr=lr, **options)


def get_scheduler(optimizer, **options):
    return get_instance(optimizer, **options)


def get_criterion(**options):
    return get_instance(**options)


def fit(model, feed, optim, criterion, penalty=None, sched=None,
        n_epochs=100, klw=1e-2, grad_clip=0., verbose=True):
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.nn.utils import clip_grad_norm_

    model.train()
    history, abort = [], False
    with tqdm.tqdm(range(n_epochs), disable=not verbose) as bar, \
            DelayedKeyboardInterrupt("ignore") as stop:
        for epoch in bar:
            epoch_loss, kl_d, grad_norm = [], 0., float("nan")
            for data, target in feed:
                optim.zero_grad()

                # get loss and penalty
                crit = criterion(model(data), target)
                if penalty is not None:
                    kl_d = penalty(model)
                loss = crit + klw * kl_d

                loss.backward()
                if grad_clip > 0:
                    grad_norm = clip_grad_norm_(model.parameters(), grad_clip)

                optim.step()
                if verbose:
                    bar.set_postfix_str(
                        f"{float(crit):.3e} {float(kl_d):.3e} |g| {grad_norm:.1e}"
                    )

                history.append((float(crit), float(kl_d)))
                epoch_loss.append(float(loss))

                # abort on nan -- no need to waste compute
                abort = np.isnan(epoch_loss[-1])
                if abort or stop:
                    break

            if abort or stop:
                break

            if sched is not None:
                # exclusions to `.step` api only apply to ReduceLROnPlateau
                if isinstance(sched, ReduceLROnPlateau):
                    sched.step(np.mean(epoch_loss))
                else:
                    sched.step()
        # end for
    # end with

    return model.eval(), history, bool(abort or stop)
