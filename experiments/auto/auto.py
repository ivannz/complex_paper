__version__ = "0.1"
import os
import copy
import tqdm
import math
import json

import torch
import numpy as np

from cplxmodule.relevance import compute_ard_masks
from cplxmodule.masks import binarize_masks

from setuptools._vendor.packaging.version import Version
from .utils import feed_limiter, feed_mover
from .utils import save_snapshot, load_snapshot
from .utils import join, deploy_optimizer


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


def run(options, folder, suffix, verbose=True):
    options = defaults(options)
    options_backup = copy.deepcopy(options)

    # placement (dtype / device) and sparsity settings
    threshold = options["threshold"]  # log(p/(1-p)), p=dropout rate
    devtype = dict(device=torch.device(options["device"]), dtype=torch.float32)

    datasets = get_datasets(options["dataset"], options["dataset_sources"])

    collate_fn = get_collate_fn(options["features"])

    feeds = get_feeds(datasets, collate_fn, options["feeds"])

    objective_terms = get_objective_terms(options["objective_terms"])

    model_factory = get_model_factory(options["model"])

    stages = options["stages"]
    model, optim, mapper = None, None, {}
    for n_stage, stage in enumerate(options["stage-order"]):
        settings = stages[stage]
        stage_backup = stage, copy.deepcopy(settings)

        # `.load_state_dict()` automatically moves to device and casts
        new_model = get_model(model_factory, settings["model"]).to(**devtype)

        # base lr is specified in the optimizer settings
        new_optim = get_optimizer(new_model, **settings["optimizer"])

        # name-id mapping for copying optimizer states
        new_mapper = {k: id(p) for k, p in new_model.named_parameters()}

        # check optimizer class and restart flag (`new_optim` exists anyway)
        restart = settings["restart"] or not isinstance(optim, new_optim.__class__)

        # continuation: inherit optimizer state and model parametrs either form
        #  the previous (hot) state or from a (cold) storage.
        optim_state = {}
        if settings["snapshot"] is not None:
            # Cold: parameters and buffers are loaded from some snapshot
            model, optim, mapper = None, None, {}
            snapshot = load_snapshot(settings["snapshot"])

            # overwrite the parameters and buffers of the model
            new_model.load_state_dict(snapshot["model"], strict=True)

            # get the saved state of the optimizer and a name-id map
            optim_state = snapshot.get("optimizer", {})
            mapper = snapshot.get("mapper", {})

            del snapshot

        elif model is not None:
            # Warm: the previous model provides the parameters
            optim_state = optim.state_dict()

            # harvest and binarize masks, and clean the related parameters
            masks = compute_ard_masks(model, hard=True, threshold=threshold)
            state_dict, masks = binarize_masks(model.state_dict(), masks)
            state_dict.update(masks)

            # Models in each stage are instances of the same underlying
            #  architecture just with differnt traits. Hence only here
            #  we allow missing or unexpected parameters when deploying
            #  the state
            new_model.load_state_dict(state_dict, strict=False)

            del state_dict, masks

        else:
            raise NotImplementedError

        # pass on the state of the optimizer
        if optim_state and not restart:
            # construct a map of id-s from `source` (left) to `target` (right)
            mapping = join(right=new_mapper, left=mapper, how="inner")
            deploy_optimizer(dict(mapping.values()),
                             source=optim_state, target=new_optim)
            del mapping
        del optim_state

        model, optim, mapper = new_model, new_optim, new_mapper

        # train
        objective = get_objective(objective_terms, settings["objective"]).to(**devtype)
        feed = FeedWrapper(feeds[stage], max=settings["n_batches_per_epoch"], **devtype)

        sched = get_scheduler(optim, **settings["lr_scheduler"])

        model.train()
        model, success, history = fit(
            model, objective, feed, optim, sched,
            n_epochs=settings["n_epochs"], grad_clip=settings["grad_clip"],
            verbose=verbose)

        # <performance assessment>
        model.eval()

        # save snapshot
        status = "" if success else "TERMINATED "
        final_snapshot = save_snapshot(
            os.path.join(folder, f"{status}{n_stage}-{stage} {suffix}.gz"),
            model=model.state_dict(),
            optimizer=optim.state_dict() if success else None,
            history=history,
            options=options_backup,
            stage=stage_backup,
            __version__=__version__
        )

        # offload model back to cpu
        model.cpu()

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
