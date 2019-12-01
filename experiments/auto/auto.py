import os
import copy
import tqdm
import math
import json

import torch
import numpy as np

from setuptools._vendor.packaging.version import Version


__version__ = "0.1"


def fit(model, objective, feed, optim, sched=None, n_epochs=100,
        grad_clip=0., verbose=True):
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.nn.utils import clip_grad_norm_
    from .delayed import DelayedKeyboardInterrupt

    history, abort = [], False
    with tqdm.tqdm(range(n_epochs), disable=not verbose) as bar, \
            DelayedKeyboardInterrupt("ignore") as stop:

        model.train()
        for epoch in bar:
            epoch_loss, grad_norm = [], float("nan")
            for data, target in feed:
                optim.zero_grad()

                # Compute the composite objective and record the components
                loss = objective(model, data, target)
                loss.backward()
                if grad_clip > 0:
                    grad_norm = clip_grad_norm_(model.parameters(), grad_clip)

                epoch_loss.append(float(loss))
                history.append((*objective.component_values_, grad_norm))

                optim.step()
                if verbose:
                    # format the components of the loss objective
                    terms = map("{:.2e}".format, history[-1][:-1])
                    status = repr(tuple(terms)).replace("'", "")

                    bar.set_postfix_str(f"{status} |g| {grad_norm:.1e}")

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

    # Collect histories of objective's components and the norm of the gradient
    *term_values, grad_norms = [np.empty(0)] * (len(objective.terms) + 1)
    if history:
        *term_values, grad_norms = map(np.array, zip(*history))

    history = dict(zip(objective.terms, term_values))
    history.update({"|g|": grad_norms})

    return model.eval(), bool(abort or stop), history
