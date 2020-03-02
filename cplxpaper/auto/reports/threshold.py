import torch

from ..utils import load_stage_snapshot, load_manifest
from ..auto import state_dict_with_masks

from .tradeoff import load_model


def load_experiment(folder):
    """Load a single experiment for the threshold report.

    Details
    -------
    Loads the model at the end of `sparsify` stage and creates an
    uninitialized model for the `fine-tune` stage, which serves as a
    receptacle for a compressed model at each threshold.
    """
    options = load_manifest(folder)

    # load 'sparsify' stage and prepare the `fine-tune`
    models = {
        "sparsify": load_model(load_stage_snapshot("sparsify", folder)),
        "fine-tune": load_model(load_stage_snapshot("fine-tune", folder))
    }

    return options, models


def evaluate_experiment(folder, *, device):
    """Evaluate an experiment."""
    device = torch.device(device)

    # get the models and the scorer
    options, models = load_experiment(folder)
    return folder, options, {}
