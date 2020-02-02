__version__ = "0.2"
import os
import copy
import tqdm
import json
import time
import warnings

import torch
import numpy as np

from cplxmodule.nn.relevance import compute_ard_masks
from cplxmodule.nn.masked import binarize_masks, deploy_masks

from scipy.special import logit
from setuptools._vendor.packaging.version import Version

from .utils import get_class, get_instance
from .utils import param_apply_map, param_defaults

from .utils import save_snapshot, load_snapshot
from .utils import join, deploy_optimizer

from .feeds import FeedMover, FeedLimiter

from .fit import fit
from .objective import ExpressionObjective, WeightedObjective

from collections import namedtuple
State = namedtuple("State", ["model", "optim", "mapper"])

benign_emergencies = (StopIteration, type(None))


def defaults(options):
    options = copy.deepcopy(options)

    # add defaults: device, threshold, and objective terms
    options = {
        "device": "cuda:0",
        "threshold": -0.5,
        "scorers": {},
        **options
    }

    assert all(key in options for key in [
        "device", "threshold", "datasets", "features", "feeds",
        "scorers", "objective_terms", "model", "stages", "stage-order"
    ])

    # check that all stages are present
    assert all(stage in options["stages"] for stage in options["stage-order"])

    # fix optional settings in stages
    for settings in options["stages"].values():
        # required: "feed", "n_epochs", "model", "objective"
        # optional: "snapshot", "restart", "reset", "grad_clip",
        #           "optimizer", "early", "lr_scheduler"

        settings.update({
            "snapshot": None,
            "restart": False,
            "reset": False,
            "grad_clip": 0.5,
            "early": None,
            "lr_scheduler": {
                "cls": "<class 'cplxpaper.musicnet.lr_scheduler.Trabelsi2017LRSchedule'>"
            },
            "optimizer": {
                "cls": "<class 'torch.optim.adam.Adam'>",
                "lr": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-08,
                "weight_decay": 0,
                "amsgrad": False
            },
            **settings
        })

        assert all(key in settings for key in [
            "snapshot", "feed", "restart", "reset", "n_epochs",
            "grad_clip", "model", "lr_scheduler", "optimizer",
            "objective", "early"
        ])

    return options


def get_datasets(recipe):
    """Get dataset instances from the recipe."""
    # "datasets"
    return {dataset: get_instance(**par) for dataset, par in recipe.items()}


def wrap_feed(feed, max_iter=-1, **devtype):
    """Return a feed that combines the limiter and the mover."""
    return FeedMover(FeedLimiter(feed, max_iter), **devtype)


def get_feature_generator(feed, recipe):
    """Create feature generator (last step before feeeding into fit)."""
    return get_instance(feed, **recipe)


def get_feeds(datasets, devtype, features, recipe):
    """Get instances of data feeds (loader with feature geenrator)."""
    # <datasets>, <devtype>, "features", "feeds"
    recipe = param_apply_map(recipe, dataset=datasets.__getitem__)

    feeds = {}
    for name, par in recipe.items():
        par = param_defaults(par, cls=str(torch.utils.data.DataLoader),
                             n_batches=-1, pin_memory=True)

        max_iter = par.pop("n_batches")
        feed = wrap_feed(get_instance(**par), max_iter=max_iter, **devtype)

        # outer wrapper is the on-the-fly feature generator
        feeds[name] = get_feature_generator(feed, features)

    return feeds


def get_scorers(feeds, recipe):
    """Get scoring objects."""
    # <feeds>, "scorers"
    recipe = param_apply_map(recipe, feed=feeds.__getitem__)
    return {name: get_instance(**par) for name, par in recipe.items()}


def compute_positive_weight(recipe):
    """Compute the +ve label weights to ensure balance given in the recipe.

    This specific to multi-output binary classification in MusicNet.
    """
    if recipe is None:
        return None

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


def get_model(recipe, **overrides):
    """Construct the instance of a model from a two-part specification."""
    # "model", "stages__*__model"
    if isinstance(overrides, dict):
        recipe = {**recipe, **overrides}  # override parameters

    return get_instance(**recipe)  # expand (shallow copy)


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


def get_early_stopper(scorers, recipe):
    """Get scheduler for the optimizer and recipe."""
    # <scorers>, "stages__*__early"
    def None_or_get_class(klass):
        return klass if klass is None else get_class(klass)

    recipe = param_apply_map(recipe, scorer=scorers.__getitem__,
                             raises=None_or_get_class)
    return get_instance(**recipe)


def state_dict_with_masks(model, **kwargs):
    """Harvest and binarize masks, and cleanup the zeroed parameters."""
    with torch.no_grad():
        masks = compute_ard_masks(model, **kwargs)
        state_dict, masks = binarize_masks(model.state_dict(), masks)

    state_dict.update(masks)
    return state_dict, masks


def state_create(recipe, stage, devtype):
    """Create a new state, i.e. model, optimizer and name-id mapper (for state
    inheritance below), from the stage and model settings.
    """
    # subsequent `.load_state_dict()` automatically moves to device and casts
    #  `model` stage in a stage are allowed to be None
    model = get_model(recipe, **stage["model"]).to(**devtype)

    # base lr is specified in the optimizer settings
    optim = get_optimizer(model, stage["optimizer"])

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
        if saved is not None:
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

        # Acquire sparsity masks and non-zero parameters
        state_dict, masks = state_dict_with_masks(old.model, **sparsity_kwargs)
        if options["reset"]:
            # Explicitly instructed to transfer masks only (if available).
            #  `reset=True` makes sure that every mask in the receiving
            #  model is initialized. A forward pass thorugh a model with
            #  an uninitialized mask would raise a RuntimeError.
            deploy_masks(state.model, state_dict=masks, reset=True)

        else:
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


def run(options, folder, suffix, verbose=True, save_optim=False):
    """The main procedure that choreographs staged training.

    Parameters
    ----------
    """
    # cudnn float32 ConvND does not seem to guarantee +ve sign of the result
    #  on inputs, that are +ve by construction.
    # torch.backends.cudnn.deterministic = True
    dttm = time.strftime("%Y%m%d-%H%M%S")

    options = defaults(options)
    options.update({
        "__version__": __version__,
        "__timestamp__": dttm,
    })

    options_backup = copy.deepcopy(options)
    with open(os.path.join(folder, "config.json"), "w") as fout:
        json.dump(options_backup, fout, indent=2, sort_keys=False)

    # placement (dtype / device)
    devtype = dict(device=torch.device(options["device"]), dtype=None)

    # The data feeds are built from three layers:
    #   Dataset -> DataLoader -> *FeedTransformer
    # `Dataset` -- physical access to data and initial transformations
    # `DataLoader` -- sampling and collating samples into batches
    # `FeedTransformer` -- host-to-device relocation or post-processing and
    #   can be chained
    # Legacy and poor development choices led fused creation of of
    #  DataLoaders and FeedTransformers
    datasets = get_datasets(options["datasets"])  # implicit dtype=float32
    feeds = get_feeds(datasets, devtype, options["features"], options["feeds"])

    # In case the datasets are imbalanced we pass then into loss
    #  objective constructor
    objective_terms = get_objective_terms(datasets, options["objective_terms"])

    # sparsity settings: threshold is log(p / (1 - p)) for p=dropout rate
    sparsity = dict(hard=True, threshold=options["threshold"])

    # scorers are objects that take a model and return its metrics on the
    #  specified feeds.
    scorers = get_scorers(feeds, options["scorers"])

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

        feed = feeds[settings["feed"]]

        formula = settings["objective"]
        objective = get_objective(objective_terms, formula).to(**devtype)

        # setup checkpointing and fit-time validation for early stopping
        # checkpointer, early_stopper = Checkpointer(...), EarlyStopper(...)
        early = None
        if "early" in settings and settings["early"] is not None:
            early = get_early_stopper(scorers, settings["early"])

        model.train()
        model, emergency, history = fit(
            model, objective, feed, optim, sched=sched, early=early,
            n_epochs=settings["n_epochs"], grad_clip=settings["grad_clip"],
            verbose=verbose)

        is_benign_emergency = isinstance(emergency, benign_emergencies)

        # Evaluate the performance on the reserved feeds, e.g. 'test*'.
        model.eval()
        performance = {}
        if is_benign_emergency:
            # Ignore warnings: no need to `.filter()` when `record=True`
            with warnings.catch_warnings(record=True):
                performance = {name: scorer(model)
                               for name, scorer in scorers.items()}

        # save snapshot
        status = "" if emergency is None else type(emergency).__name__
        status += " " if status else ""
        final_snapshot = save_snapshot(
            os.path.join(folder, f"{n_stage}-{stage} {status}{suffix}.gz"),

            # state
            model=state.model.state_dict(),
            optim=dict(
                cls=str(type(state.optim)),
                state=state.optim.state_dict()
            ) if (is_benign_emergency and save_optim) else None,
            mapper=state.mapper if (is_benign_emergency and save_optim) else None,

            # meta data
            history=history,
            performance=performance,
            early_history=np.array(getattr(early, "history_", [])),
            options=options_backup,
            stage=stage_backup,

            __version__=__version__
        )

        # offload the model back to the cpu
        model.cpu()
        snapshots.append(final_snapshot)

        # emergency termination: cannot continue
        if not is_benign_emergency:
            break

    return snapshots
