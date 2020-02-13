import re
import gc
import torch

from functools import partial

from .utils import get_datasets, dict_get_one

from .. import auto
from ..utils import load_stage_snapshot, load_manifest

from ...mnist.performance import MNISTBasePerformance
from ...musicnet.performance import MusicNetBasePerformance


def load_model(snapshot, errors="ignore"):
    """Recover the model from the snapshot."""
    if errors not in ("ignore", "raise"):
        raise ValueError(f"`errors` must be either 'ignore' or 'raise'.")

    if any(k not in snapshot for k in ["options", "stage", "model"]):
        if errors == "raise":
            raise ValueError("Bad snapshot.")
        return torch.nn.Module()

    options = snapshot["options"]
    _, settings = snapshot["stage"]

    model = auto.get_model(options["model"], **settings["model"])
    model.to(device=torch.device("cpu"))
    model.load_state_dict(snapshot["model"])

    return model


def load_experiment(folder):
    """Load a single experiment for trade-off report.

    Details
    -------
    Loads the models at the end of `dense` and `fine-tune` stages. And recovers
    the model that existed just before the `fine-tune` stage from `sparsify`
    and the sparsity `threshold`, specified in the experiment.
    """
    options = load_manifest(folder)

    # load 'dense'
    models = {"dense": load_model(load_stage_snapshot("dense", folder))}

    # "post-fine-tune"
    snapshot = load_stage_snapshot("fine-tune", folder)
    models["post-fine-tune"] = load_model(snapshot)

    # "pre-fine-tune": load model from `fine-tune` and deploy the masks
    #  and weights onto it from `sparsify` using the prescribed threshold.
    state_dict, masks = auto.state_dict_with_masks(
        load_model(load_stage_snapshot("sparsify", folder)),
        hard=True, threshold=options["threshold"])

    models["pre-fine-tune"] = load_model(snapshot)
    models["pre-fine-tune"].load_state_dict(state_dict, strict=False)

    return options, models


def get_scorers(options, **devtype):
    """Creat scoring objects suitable for the experiment.

    Details
    -------
    Excludes all datasets the tag of whoch contains the word `train`.
    Attemps auto-detect performance scoring objects from the dataset name.
    """

    # use threshold set in the experiment
    threshold = options["threshold"]

    # remove any feed that is tagged as train
    feeds_recipes = {tag: feed for tag, feed in options["feeds"].items()
                     if "train" not in tag}

    dataset_tags = set(feed["dataset"] for feed in feeds_recipes.values())
    if not dataset_tags:
        raise ValueError(f"Unable to find any non-train datasets")

    # auto detect the performance object from the dataset
    cls = options["datasets"][next(iter(dataset_tags))]["cls"]
    match = re.match(r"^<class 'cplxpaper\.([^\.]+)\.dataset\.(.*?)'>$", cls)
    if match is None:
        raise ValueError(f"Unrecognized dataset `{cls}`")

    submod, obj = map(str.lower,  match.groups())

    # MusciNet
    if submod in ("musicnet",):
        # MusciNet
        Performance = partial(MusicNetBasePerformance, curves=False)

    elif submod in ("mnist", "cifar"):
        # MNIST-like or CIFAR10/100
        Performance = MNISTBasePerformance

    else:
        raise ValueError(f"Unknown dataset submodule `{submod}` for `{obj}`.")

    # load the datasets which are used in the remaining non-train feeds
    feeds = auto.get_feeds(get_datasets({
        tag: dataset for tag, dataset in options["datasets"].items()
        if tag in dataset_tags
    }), devtype, options["features"], feeds_recipes)

    return {tag: Performance(feed, threshold=threshold)
            for tag, feed in feeds.items()}


def score_models(scorer, models, **devtype):
    """Score the given models with the specified object."""
    scores = []
    for name, model in models.items():
        model.to(**devtype)
        with torch.no_grad():
            scores.append((name, scorer(model.eval())))
        model.cpu()

    return dict(scores)


# MNIST-like (tradeoff, INCOMPATIBLE),  MusicNet (tradeoff, INCOMPATIBLE)
def evaluate_experiment(folder, *, device):
    """Evaluate an experiment."""
    device = torch.device(device)

    # get the models and the scorer
    options, models = load_experiment(folder)
    scorers = get_scorers(options, device=device)

    # score each model on device
    scores = []
    for name, scorer in scorers.items():
        scores.append((name, score_models(scorer, models, device=device)))

    # this might help with creeping gpu memory
    del models, scorers

    gc.collect()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    return folder, options, dict(scores)
