__version__ = "0.2"
import os
import gc
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
    """Fill in the omitted specifications."""
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
    """Get dataset instances from the recipe.

    Parameters
    ----------
    recipe : dict of dict
        The specifications of the core datasets.

    Returns
    -------
    datasets : dict of datasets
        A dictionary of dataset instances, compatible with torch's
        DataLoader objects.
    """
    # "datasets"
    return {dataset: get_instance(**par) for dataset, par in recipe.items()}


def wrap_feed(feed, max_iter=-1, **devtype):
    """Return a feed that combines the limiter and the mover.

    Parameters
    ----------
    feed : DataLoader-like object
        The data feed, the batches of which should be moved to the device.

    max_iter : int, default=-1
        The upper limit on the number of data batches provided by the feed.

    **devtype : dict
        The dictionary specifying the device and dtype conversion for pytorch.

    Returns
    -------
    feed : DataLoader-like object
        A wrapped feed.
    """
    return FeedMover(FeedLimiter(feed, max_iter), **devtype)


def get_feature_generator(feed, recipe):
    """Create feature generator (last step before feeding into fit).

    Parameters
    ----------
    feed : torch.optim
        The data feed to create the specified feature transformation for.

    recipe : dict
        The specifications of the feature transformation.

    Returns
    -------
    feed : object
        A wrapped feed.
    """
    return get_instance(feed, **recipe)


def get_feeds(datasets, devtype, features, recipe):
    """Get instances of data feeds (loader with feature generator).

    Parameters
    ----------
    datasets : dict of Dataset objects
        The source datasets over which to construct and package data loaders.

    devtype : dict
        The dictionary specifying the device and dtype conversion for pytorch.

    features : dict
        Teh dictionary, specifying the feature transformations.

    recipe : dict of dict
        The nested dictionary of data feed specifications.

    Returns
    -------
    feeds : dict of object
        The properly packaged DataLoaders.
    """

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
    """Get scoring objects.

    Parameters
    ----------
    feeds : dict of DataLoader objects
        The ready-for-use data feeds for scorers to use.

    recipe : dict
        The nested dictionary of scorer specifications.

    Returns
    -------
    scorers : dict of object
        The properly named instances of scoring objects.
    """

    # <feeds>, "scorers"
    recipe = param_apply_map(recipe, feed=feeds.__getitem__)
    return {name: get_instance(**par) for name, par in recipe.items()}


def compute_positive_weight(recipe):
    """Compute the +ve label weights to ensure balance given in the recipe.

    Parameters
    ----------
    recipe : dict
        The dictionary of parameters specifying class rebalance and the
        dataset, used to gather the label statistics from.

    Returns
    -------
    pos_weight : tensor
        The weights of positive label in multi-output binary classification.

    Details
    -------
    This specific to multi-output binary classification in MusicNet, but never
    actually used.
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
    """Gather the components of the objective function.

    Parameters
    ----------
    datasets : dict of datasets
        The datasets to refer, used to estimated the class weights for
        imbalanced classification (seldom used).

    recipe : dict
        The nested dictionary of specifications of the loss and kl-divergence
        and other terms in the composite objective.

    Returns
    -------
    objectives : dict of object
        The properly named instances of the terms in the objective function.
    """

    # <datasets>, "objective_terms"
    recipe = param_apply_map(recipe, dataset=datasets.__getitem__)

    objectives = {}
    for name, par in recipe.items():
        par = param_apply_map(par, pos_weight=compute_positive_weight)
        objectives[name] = get_instance(**par)

    return objectives


def get_model(recipe, **overrides):
    """Construct the instance of a model from a two-part specification.

    Parameters
    ----------
    recipe : dict
        The base recipe for the model.

    **overrides : dict
        Overrides or extra parameters for the model's recipe.

    Returns
    -------
    model : torch.nn.Module
        The instance of the model, created from the provided specifications
        and overrides.
    """

    # "model", "stages__*__model"
    if isinstance(overrides, dict):
        recipe = {**recipe, **overrides}  # override parameters

    return get_instance(**recipe)  # expand (shallow copy)


def get_objective(objective_terms, recipe):
    """Construct the objective function from the recipe and components.

    Parameters
    ----------
    objective_terms : dict of object
        The loss and other terms (named), that should be used in the composite
        objective specified by the recipe.

    recipe : dict, or str
        The specifications of the objective. Can be either a dictionary, which
        specifies the weights of each term in the sum, or a pythonic arithmetic
        expression, that uses terms' names as variables.

    Returns
    -------
    objective : object
        An instance of the optimization objective.
    """

    # <objective_terms>, "stages__*__objective"
    if isinstance(recipe, dict):
        return WeightedObjective(recipe, **objective_terms)

    if isinstance(recipe, str):
        return ExpressionObjective(recipe, **objective_terms)


def get_optimizer(module, recipe):
    """Get the optimizer for the module from the recipe.

    Parameters
    ----------
    module : torch.optim
        The module, to parameters of which an optimizer should be attached to.

    recipe : dict
        The specifications of the optimizer.

    Returns
    -------
    optim : torch.optim
        An instance of an optimizer.
    """

    # <module>, "stages__*__optimizer"
    return get_instance(module.parameters(), **recipe)


def get_scheduler(optimizer, recipe):
    """Get scheduler for the optimizer and recipe.

    Parameters
    ----------
    optimizer : torch.optim
        The optimizer to attach the learning rate scheduler to.

    recipe : dict
        The specifications of the scheduler.

    Returns
    -------
    scorer : torch.optim.lr_scheduler
        An instance of a scheduler.
    """

    # <optimizer>, "stages__*__lr_scheduler"
    return get_instance(optimizer, **recipe)


def get_early_stopper(scorers, recipe):
    """Get scheduler for the optimizer and recipe.

    Parameters
    ----------
    scorers : dict
        A dictionary of holdout set scoring object instances.

    recipe : dict
        The specification of the early stopper.

    Returns
    -------
    scorer : object
        An instance of a ready-for-use scorer.
    """

    # <scorers>, "stages__*__early"
    def None_or_get_class(klass):
        return klass if klass is None else get_class(klass)

    recipe = param_apply_map(recipe, scorer=scorers.__getitem__,
                             raises=None_or_get_class)
    return get_instance(**recipe)


def state_dict_with_masks(model, **kwargs):
    """Harvest and binarize masks, and cleanup the zeroed parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model to collect masks and sate dict from.

    **sparsity_kwargs : dict
        The keyword arguments passed to sparsification procedures to gather
        the binary dropout masks.

    Returns
    -------
    state_dict : dict
        The dictionary of parameter values, buffers and masks for loading with
        torch's api.

    mask : dict
        The same dict as `state_dict` but with the sparsity masks only (for
        convenience).
    """

    with torch.no_grad():
        masks = compute_ard_masks(model, **kwargs)
        state_dict, masks = binarize_masks(model.state_dict(), masks)

    state_dict.update(masks)
    return state_dict, masks


def state_create(recipe, stage, devtype):
    """Create a new state, i.e. model, optimizer and name-id mapper (for state
    inheritance below), from the stage and model settings.

    Parameters
    ----------
    recipe : dict
        The base recipe for the model.

    stage : dict
        A dictionary specifying the optimizer and overrides for the model's
        recipe.

    devtype : dict
        The dictionary specifying the device and dtype conversion for pytorch.

    Returns
    -------
    state : State
        A named tuple with an instance of the model, an optimizer associated
        with its parameters and special name-id mapping used when transferring
        internal state between optimizers of different models.
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

    Parameters
    ----------
    state : State
        A named tuple keeping together the model, the optimizer associated with
        it, and a layer name - parameter id correspondence, used to facilitate
        transfer of state between models, that share a common core set of
        parameters, e.g. when transferring weight's accumulated gradient stats
        from ordinary linear models to Bayesian layers. This occasionally helps
        warm start the SGD.

    options : dict
        The key `reset` specifies whether the model's parameters should be
        transferred from the `old` state, `restart` -- if the optimizer's
        internal should be copied, and `snapshot` -- the path to a snapshot
        for cold start.

    old : dict
        Explicit keyword argument specifying the state to inherit parameters
        form.

    **sparsity_kwargs : dict
        The keyword arguments passed to sparsification procedures.

    Returns
    -------
    state : State
        The updated model, optimizer and mapping.
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
            #  model is initialized. A forward pass through a model with
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
    options : dict
        The manifest of an experiment: a nested dictionary, specifying the
        datasets, batch loaders, models, optimizers, schedulers, stages, and
        coefficients in the objective.

    folder : str
        The path at which to save the experiment: a copy of the manifest in
        `config.json` and snapshot of the model by the end of each training
        stage.

    suffix : str
        An tag appended to the name of each snapshot.

    verbose : bool, default=True
        A flag indicating if a progress bar with running diagnostic information
        is to be printed.

    save_optim : bool, default=False
        A flag specifying if the optimizer's internal state should be a part
        of the recorded snapshot. Useful for cold start or continuation of
        experiment's stages run earlier.
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
    device = torch.device(options["device"])
    devtype = dict(device=device, dtype=None)

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
        n_epochs = settings["n_epochs"]

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
            n_epochs=n_epochs, grad_clip=settings["grad_clip"],
            verbose=verbose)

        is_benign_emergency = isinstance(emergency, benign_emergencies)

        # Evaluate the performance on the reserved feeds, e.g. 'test*'.
        model.eval()
        performance = {}
        if is_benign_emergency and n_epochs > 0:
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

    # this might help with creeping gpu memory
    gc.collect()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    return snapshots


def debug(options, folder, suffix, verbose=True, save_optim=False):
    """For the purposes of debugging randomness and GPU memory."""
    device = torch.device(options["device"])
    values = torch.randn(1024, device=device)

    final_snapshot = save_snapshot(
        os.path.join(folder, f"blob{suffix}.gz"),
        values=values.cpu()
    )

    # this might help with creeping gpu memory
    gc.collect()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    return final_snapshot
