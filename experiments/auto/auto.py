__version__ = "0.1"
import os
import copy
import tqdm
import math
import json

import torch
import numpy as np

from setuptools._vendor.packaging.version import Version
from .utils import feed_limiter, feed_mover


def defaults(options):
    # currently no defaults
    return copy.deepcopy(options)


def get_datasets(factory, recipe):
    """Get dataset instances from the factory specification and recipe."""
    # "dataset", "dataset_sources"
    raise NotImplementedError


def get_collate_fn(recipe):
    """Get collation function responsible for batch-feature generation."""
    # "features"
    raise NotImplementedError


def get_feeds(datasets, collate_fn, recipe):
    """Get instances of data loaders from the datasets and collate function."""
    # <datasets>, <collate_fn>, "feeds"
    raise NotImplementedError


def get_objective_terms(recipe):
    """Gather the components of the objective function."""
    # "objective_terms"
    raise NotImplementedError


def get_model_factory(recipe):
    """Partial constructor of the model. Assumes a particual common API."""
    # "model"
    raise NotImplementedError


def get_model(factory, recipe):
    """Partial constructor of the model. Assumes a particual common API."""
    # <factory>, "stages__*__model"
    raise NotImplementedError


def get_objective(objective_terms, recipe):
    """Construct the objective function from the recipe and components."""
    # <objective_terms>, "stages__*__objective"
    raise NotImplementedError


def get_optimizer(module, *, lr, **kwargs):
    """Get the optimizer for the module from explicit lr and kwargs."""
    raise NotImplementedError


def get_scheduler(optimizer, **kwargs):
    """Get scheduler for the optimizer and kwargs."""
    raise NotImplementedError


class FeedWrapper(object):
    def __init__(self, feed, max=1000, **kwargs):
        self.feed, self.max, self.kwargs = feed, max, kwargs

    def __iter__(self):
        feed = feed_limiter(self.feed, self.max)
        yield from feed_mover(feed, **self.kwargs)

    def __len__(self):
        return len(self.feed)


def run(options, verbose=True):
    options = defaults(options)

    # placement (dtype / device) and sparsity settings
    threshold = options["threshold"]  # log(p/(1-p)), p=dropout rate
    devtype = dict(device=torch.device(options["device"]), dtype=torch.float32)

    datasets = get_datasets(options["dataset"], options["dataset_sources"])

    collate_fn = get_collate_fn(options["features"])

    feeds = get_feeds(datasets, collate_fn, options["feeds"])

    objective_terms = get_objective_terms(options["objective_terms"])

    model_factory = get_model_factory(options["model"])

    model, optim = None, None
    for stage, settings in options["stages"].itmes():
        new_model = get_model(model_factory, settings["model"]).to(**devtype)
        new_optim = get_optimizer(new_model, **settings["optimizer"])

        # <stage continuation>
        model, optim = new_model, new_optim

        # train
        objective = get_objective(objective_terms, settings["objective"]).to(**devtype)
        feed = FeedWrapper(feeds[stage], max=settings["n_batches_per_epoch"], **devtype)

        sched = get_scheduler(optim, **settings["lr_scheduler"])

        model.train()
        model, state, history = fit(
            model, objective, feed, optim, sched,
            n_epochs=settings["n_epochs"], grad_clip=settings["grad_clip"],
            verbose=verbose)

    return


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
