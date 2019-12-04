__version__ = "0.1"
import os
import copy
import tqdm
import math
import json
import warnings

import torch
import numpy as np

from cplxmodule.relevance import compute_ard_masks
from cplxmodule.masked import binarize_masks

from scipy.special import logit
from setuptools._vendor.packaging.version import Version

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from .delayed import DelayedKeyboardInterrupt

from .performance import evaluate

from .utils import get_class, get_factory, get_instance
from .utils import param_apply_map, param_defaults

from .utils import FeedMover, FeedLimiter
from .utils import save_snapshot, load_snapshot
from .utils import join, deploy_optimizer

from .objective import ExpressionObjective, WeightedObjective

from collections import namedtuple
State = namedtuple("State", ["model", "optim", "mapper"])


def defaults(options):
    # currently no defaults
    return copy.deepcopy(options)


def get_datasets(factory, recipe):
    """Get dataset instances from the factory specification and recipe."""
    # "dataset", "dataset_sources"
    factory = get_factory(**factory)
    return {name: factory(**source)
            for name, source in recipe.items()}


def get_collate_fn(recipe):
    """Get collation function responsible for batch-feature generation."""
    # "features"
    return get_instance(**recipe)


def get_feeds(datasets, collate_fn, recipe):
    """Get instances of data loaders from the datasets and collate function."""
    # <datasets>, <collate_fn>, "feeds"
    recipe = param_apply_map(recipe, dataset=datasets.__getitem__)

    feeds = {}
    for name, par in recipe.items():
        par = param_defaults(par, pin_memory=True)
        feeds[name] = torch.utils.data.DataLoader(**par, collate_fn=collate_fn)

    return feeds


def wrap_feed(feed, max_iter=-1, **devtype):
    """Return a feed that combines the limiter and the mover."""
    return FeedMover(FeedLimiter(feed, max_iter), **devtype)


def get_special_feeds(feeds, devtype, special):
    """Prepare feeds for special activities, like testing performance."""
    special_feeds = {}
    for name, feed in feeds.items():
        if name in special:
            special_feeds[name] = wrap_feed(feed, max_iter=-1, **devtype)

    # may be empty
    return special_feeds


def compute_positive_weight(recipe):
    """Compute the +ve label weights to ensure balance given in the recipe."""
    # "pos_weight" (with mapped dataset)
    recipe = param_defaults(recipe, alpha=0.5, max=1e2)

    # average across pieces of data (axis 0 in probabilities)
    proba_hat = recipe["dataset"].probability.mean(axis=0)

    # clip to within (0, 1) to prevent saturation
    proba_hat_clipped = proba_hat.clip(1e-6, 1 - 1e-6)

    # w_+ = \alpha (\tfrac{m}{n_+} - 1) = \alpha \tfrac{1 - p_+}{p_+}
    pos_weight = np.exp(- logit(proba_hat_clipped)) * recipe["alpha"]

    # clip weights to avoid overflows
    pos_weight_clipped = pos_weight.clip(max=recipe["max"])

    return torch.from_numpy(pos_weight_clipped).float()


def get_objective_terms(datasets, recipe):
    """Gather the components of the objective function."""
    # <datasets>, "objective_terms"
    recipe = param_apply_map(recipe, dataset=datasets.__getitem__)

    objectives = {}
    for name, par in recipe.items():
        par = param_apply_map(par, pos_weight=compute_positive_weight)
        objectives[name] = get_instance(**par)

    return objectives


def get_model(factory, recipe):
    """Construct the instance of a model from a two-part specification."""
    # "model", "stages__*__model"
    return get_instance(**factory, **recipe)


def get_objective(objective_terms, recipe):
    """Construct the objective function from the recipe and components."""
    # <objective_terms>, "stages__*__objective"
    if isinstance(recipe, dict):
        return WeightedObjective(recipe, **objective_terms)

    if isinstance(recipe, str):
        return ExpressionObjective(recipe, **objective_terms)


def get_optimizer(module, recipe):
    """Get the optimizer for the module from the recipe."""
    # <module>, "stages__*__optimizer"
    return get_instance(module.parameters(), **recipe)


def get_scheduler(optimizer, recipe):
    """Get scheduler for the optimizer and recipe."""
    # <optimizer>, "stages__*__lr_scheduler"
    return get_instance(optimizer, **recipe)


def state_create(factory, settings, devtype):
    """Create a new state, i.e. model, optimizer and name-id mapper (for state
    inheritance below), from the settings and model factory.
    """

    # subsequent `.load_state_dict()` automatically moves to device and casts
    model = get_model(factory, settings["model"]).to(**devtype)

    # base lr is specified in the optimizer settings
    optim = get_optimizer(model, settings["optimizer"])

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

    else:  # snapshot is None and old.model is None
        pass

    # check optimizer inheritance and restart flag, and deploy the state
    if (optim_state and inheritable) and not options["restart"]:
        # construct a map of id-s from `source` (left) to `target` (right)
        mapping = join(right=state.mapper, left=source_mapper, how="inner")
        deploy_optimizer(dict(mapping.values()), source=optim_state,
                         target=state.optim)
        del mapping

    del optim_state, source_mapper

    return state


def state_dict_to_cpu(state_dict):
    # .state_dict() references tensors, so we copy .data to CPU
    return {key: par.data.cpu() for key, par in state_dict.items()}


def run(options, folder, suffix, verbose=True):
    """The main procedure that choreographs staged training.

    Parameters
    ----------
    """
    options = defaults(options)
    options_backup = copy.deepcopy(options)

    # placement (dtype / device)
    devtype = dict(device=torch.device(options["device"]), dtype=torch.float32)

    # sparsity settings: threshold is log(p / (1 - p)) for p=dropout rate
    sparsity = dict(hard=True, threshold=options["threshold"])

    datasets = get_datasets(options["dataset"], options["dataset_sources"])

    collate_fn = get_collate_fn(options["features"])

    feeds = get_feeds(datasets, collate_fn, options["feeds"])
    special_feeds = get_special_feeds(feeds, devtype, ("test", "valid"))

    objective_terms = get_objective_terms(datasets, options["objective_terms"])

    snapshots = []
    stages, state = options["stages"], State(None, None, {})
    for n_stage, stage in enumerate(options["stage-order"]):
        settings = stages[stage]
        stage_backup = stage, copy.deepcopy(settings)

        new = state_create(options["model"], settings, devtype)
        state = state_inherit(new, settings, old=state, **sparsity)

        # unpack the state, initialize objective and scheduler, and train
        model, optim = state.model, state.optim
        sched = get_scheduler(optim, settings["lr_scheduler"])

        name, max_iter = settings["feed"], settings["n_batches_per_epoch"]
        feed = wrap_feed(feeds[name], max_iter=max_iter, **devtype)

        formula = settings["objective"]
        objective = get_objective(objective_terms, formula).to(**devtype)

        # setup checkpointing and fit-time validation for early stopping
        # checkpointer, early_stopper = Checkpointer(...), EarlyStopper(...)

        model.train()
        model, emergency, history = fit(
            model, objective, feed, optim, sched,
            n_epochs=settings["n_epochs"], grad_clip=settings["grad_clip"],
            verbose=verbose)

        # Evaluate the performance on the reserved feeds, e.g. `test`.
        model.eval()
        performance = {}
        if not emergency:
            # Ignore warnings: no need to `.filter()` when `record=True`
            with warnings.catch_warnings(record=True):
                performance = {name: evaluate(model, feed, curves=False)
                               for name, feed in special_feeds.items()}

        # save snapshot
        status = "TERMINATED " if emergency else ""
        final_snapshot = save_snapshot(
            os.path.join(folder, f"{status}{n_stage}-{stage} {suffix}.gz"),

            # state
            model=state.model.state_dict(),
            optim=dict(
                cls=str(type(state.optim)),
                state=state.optim.state_dict()
            ) if not emergency else None,
            mapper=state.mapper if not emergency else None,

            # meta data
            history=history,
            performance=performance,
            options=options_backup,
            stage=stage_backup,

            __version__=__version__
        )

        # offload the model back to the cpu
        model.cpu()
        snapshots.append(final_snapshot)

        # emergency termination: cannot continue
        if emergency:
            break

    return snapshots


class TerminateFit(Exception):
    pass


def scheduler_step(sched, value):
    if sched is None:
        return

    # exclusions to `.step` api only apply to ReduceLROnPlateau
    if isinstance(sched, ReduceLROnPlateau):
        sched.step(value)
    else:
        sched.step()


def fit(model, objective, feed, optim, sched=None, n_epochs=100,
        grad_clip=0., verbose=True):
    """Fit a model to the objective on the data feed for specified number of
    epochs with optimizer, lr-schedule and gradient clipping.

    Parameters
    ----------

    Returns
    -------

    Details
    -------
    Forces the model in `train` mode before the nested SGD loop and forces it
    into `eval` mode afterwards.
    """
    history, emergency = [], False
    with tqdm.tqdm(range(n_epochs), disable=not verbose) as bar, \
            DelayedKeyboardInterrupt("ignore") as stop:

        model.train()
        try:
            for epoch in bar:
                epoch_loss, grad_norm = [], float("nan")
                for data, target in feed:
                    backup = state_dict_to_cpu(model.state_dict())

                    # (closure) "The closure should clear the gradients,
                    #  compute the loss and gradients, and return the loss."
                    #  https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
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
                    if np.isnan(epoch_loss[-1]) or bool(stop):
                        raise TerminateFit

                scheduler_step(sched, np.mean(epoch_loss))
                # checkpointer and earlystopper steps

        except TerminateFit:
            if np.isnan(epoch_loss[-1]):
                model.load_state_dict(backup)

            emergency = True
    # end with

    # Collect histories of objective's components and the norm of the gradient
    *term_values, grad_norms = [np.empty(0)] * (len(objective.terms) + 1)
    if history:
        *term_values, grad_norms = map(np.array, zip(*history))

    history = dict(zip(objective.terms, term_values))
    history.update({"|g|": grad_norms})

    return model.eval(), emergency, history
