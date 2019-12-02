__version__ = "0.1"
import os
import copy
import tqdm
import math
import json

import torch
import numpy as np

from cplxmodule.relevance import compute_ard_masks
from cplxmodule.masked import binarize_masks

from setuptools._vendor.packaging.version import Version

from .utils import get_class
from .utils import feed_limiter, feed_mover
from .utils import save_snapshot, load_snapshot
from .utils import join, deploy_optimizer

from collections import namedtuple
State = namedtuple("State", ["model", "optim", "mapper"])


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


def get_objective_terms(datasets, recipe):
    """Gather the components of the objective function."""
    # "objective_terms"
    raise NotImplementedError


def get_model_factory(recipe):
    """Partial constructor of the model. Assumes a particular common API."""
    # "model"
    raise NotImplementedError


def get_model(factory, recipe):
    """Partial constructor of the model. Assumes a particular common API."""
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
    """A class that combines the limiter and the mover."""
    def __init__(self, feed, max=1000, **kwargs):
        self.feed, self.max, self.kwargs = feed, max, kwargs

    def __iter__(self):
        feed = feed_limiter(self.feed, self.max)
        yield from feed_mover(feed, **self.kwargs)

    def __len__(self):
        return len(self.feed)


def state_create(factory, settings, devtype):
    """Create a new state, i.e. model, optimizer and name-id mapper (for state
    inheritance below), from the settings and model factory.
    """

    # (note) `.load_state_dict()` automatically moves to device and casts
    model = get_model(factory, settings["model"]).to(**devtype)

    # base lr is specified in the optimizer settings
    optim = get_optimizer(model, **settings["optimizer"])

    # name-id mapping for copying optimizer states
    mapper = {k: id(p) for k, p in model.named_parameters()}

    return State(model, optim, mapper)


def state_inherit(state, options, *, old=None, **sparsity_kwargs):
    """Inherit optimizer state and model parameters either form
    the previous (hot) state or from a (cold) storage.
    """
    # model state is inherited anyway, optimizer is conditional.
    optim_state, source_mapper, inheritable = {}, {}, False
    if options["snapshot"] is not None:
        # Cold: parameters and buffers are loaded from some snapshot
        snapshot = load_snapshot(options["snapshot"])

        # get the saved state of the optimizer and a name-id map
        saved = snapshot.get("optim", {})
        optim_state = saved.get("state", {})
        source_mapper = snapshot.get("mapper", {})

        # see if stored state an state.optim's state are compatible
        inheritable = isinstance(state.optim, get_class(saved.get("cls")))

        # overwrite the parameters and buffers of the model
        state.model.load_state_dict(snapshot["model"], strict=True)

        del snapshot, saved

    elif isinstance(old, State) and old.model is not None:
        # Warm: the previous model provides the parameters
        optim_state, source_mapper = old.optim.state_dict(), old.mapper
        inheritable = isinstance(state.optim, type(old.optim))

        # Harvest and binarize masks, and clean the related parameters
        masks = compute_ard_masks(old.model, **sparsity_kwargs)
        state_dict, masks = binarize_masks(old.model.state_dict(), masks)
        state_dict.update(masks)

        # Models in each stage are instances of the same underlying
        #  architecture just with different traits. Hence only here
        #  we allow missing or unexpected parameters when deploying
        #  the state
        state.model.load_state_dict(state_dict, strict=False)

        del state_dict, masks

    else:
        # no snapshot, or old is None, or old is State, but old.model is None
        raise NotImplementedError

    # check optimizer inheritance and restart flag, and deploy the state
    if (optim_state and inheritable) and not options["restart"]:
        # construct a map of id-s from `source` (left) to `target` (right)
        mapping = join(right=state.mapper, left=source_mapper, how="inner")
        deploy_optimizer(dict(mapping.values()), source=optim_state,
                         target=state.optim)
        del mapping

    del optim_state, source_mapper

    return state


def evaluate(model, feed):
    raise NotImplementedError


def run(options, folder, suffix, verbose=True):
    options = defaults(options)
    options_backup = copy.deepcopy(options)

    # placement (dtype / device)
    devtype = dict(device=torch.device(options["device"]), dtype=torch.float32)

    # sparsity settings: threshold is log(p / (1 - p)) for p=dropout rate
    sparsity = dict(hard=True, threshold=options["threshold"])

    datasets = get_datasets(options["dataset"], options["dataset_sources"])

    collate_fn = get_collate_fn(options["features"])

    feeds = get_feeds(datasets, collate_fn, options["feeds"])

    objective_terms = get_objective_terms(datasets, options["objective_terms"])

    model_factory = get_model_factory(options["model"])

    stages, state = options["stages"], State(None, None, {})
    for n_stage, stage in enumerate(options["stage-order"]):
        settings = stages[stage]
        stage_backup = stage, copy.deepcopy(settings)

        state = state_inherit(state_create(model_factory, settings, devtype),
                              settings, old=state, **sparsity)

        # unpack the state, initialize objective and scheduler, and train
        model, optim = state.model, state.optim

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
        test_performance = evaluate(model, ...)

        # save snapshot
        status = "" if success else "TERMINATED "
        final_snapshot = save_snapshot(
            os.path.join(folder, f"{status}{n_stage}-{stage} {suffix}.gz"),
            # state
            model=state.model.state_dict(),
            optim=dict(
                cls=str(type(state.optim)),
                state=state.optim.state_dict()
            ) if success else None,
            mapper=state.mapper if success else None,
            # meta data
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
