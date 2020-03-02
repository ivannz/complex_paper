import gc
import torch

from ..utils import load_stage_snapshot, load_manifest
from ..auto import state_dict_with_masks

from .tradeoff import load_model, get_scorers


def load_experiment(folder):
    """Load a single experiment for the threshold report.

    Details
    -------
    Loads the model at the end of `sparsify` stage and creates an
    uninitialized model for the `fine-tune` stage, which serves as a
    receptacle for a compressed model at each threshold.
    """
    options = load_manifest(folder)

    # load 'sparsify' stage and prepare the maskable version of the model
    models = {
        "sparsify": load_model(load_stage_snapshot("sparsify", folder)),
        "masked": load_model(load_stage_snapshot("fine-tune", folder))
    }

    return options, models


def score_model(scorer, models, threshold, **devtype):
    """Score the given models with the specified object."""
    model = models["masked"].to(**devtype)
    with torch.no_grad():
        # get the masks
        state_dict, masks = state_dict_with_masks(
            models["sparsify"], threshold=threshold, hard=True)

        # copy weights and deploy masks onto the next-stage model
        model.load_state_dict(state_dict, strict=False)
        scores = scorer(model.eval())

    model.cpu()
    return scores


def evaluate_experiment(folder, threshold, *, device):
    """Evaluate an experiment."""
    device = torch.device(device)

    # get the models and the scorer
    options, models = load_experiment(folder)

    scorers = get_scorers(options, device=device)

    # score each model on device
    scores = []
    for name, scorer in scorers.items():
        scores.append((
            name, score_model(scorer, models, threshold, device=device)
        ))

    options["threshold"] = threshold

    # this might help with creeping gpu memory
    del models, scorers

    gc.collect()
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    return folder, threshold, options, dict(scores)
