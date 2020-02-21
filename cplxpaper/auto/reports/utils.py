import os
import re
import pickle

from functools import lru_cache
from ..utils import verify_experiment

from .. import auto


def restore(report):
    """Recover sequentially pickled objects from a binary file.

    Details
    -------
    Pickled objects can be concatenated, and unpickling extracts one
    object at a time and advances the stream location.
    """
    with open(report, "rb") as storage:
        while True:
            try:
                yield pickle.load(storage)

            except EOFError:
                break


def enumerate_experiments(grid):
    """Iterate over all complete experiments in a grid."""
    grid = os.path.abspath(os.path.normpath(grid))
    grid, _, filenames = next(os.walk(grid), ((), (), ()))
    for name, ext in map(os.path.splitext, filenames):
        if ext != ".json" or name.startswith("."):
            continue

        experiment = os.path.join(grid, name)
        if not verify_experiment(experiment):
            continue

        yield experiment


@lru_cache(None)
def _get_datasets(key):
    return auto.get_datasets(pickle.loads(key))


def get_datasets(datasets):
    """A dirty hack to avoid loading the same dataset over and over."""
    return _get_datasets(pickle.dumps(datasets))


def dict_get_one(d, *keys):
    for k in keys:
        if k in d:
            return d[k]

    raise KeyError


def get_model_tag(opt):
    # extract the class name
    cls = opt["stages__sparsify__model__cls"]

    pat = re.compile(r"^<class '.*\.models.*\.((?:real|complex)\..+)'>$")
    cls = pat.sub(r"\1", cls)

    # get the model kind: real/complex
    if not cls.startswith(("real.", "complex.")):
        raise ValueError("Unknown model type.")

    if cls.startswith("real."):
        kind, cls = "R", cls[5:]
    elif cls.startswith("complex."):
        kind, cls = "C", cls[8:]

    # handle real `double` and cplx `half`
    if kind == "R" and opt.get("model__double", False):
        kind = kind + "*2"

    elif kind == "C" and opt.get("model__half", False):
        kind = kind + "/2"

    # get method
    if not cls.endswith(("VD", "ARD")):
        raise ValueError("Unknown Bayesian method.")

    if cls.endswith("VD"):
        method, cls = "VD", cls[:-2]
    elif cls.endswith("ARD"):
        method, cls = "ARD", cls[:-3]

    return {"model": cls, "kind": kind, "method": method}


def get_dataset(opt):
    cls = dict_get_one(opt, "datasets__musicnet-test-128__cls",
                            "datasets__test__cls")
    assert cls is not None

    pat = re.compile(r"^<class '.*?\.(?:mnist|musicnet|cifar)\.dataset\.(.*?)'>$")
    cls = pat.sub(r"\1", cls).lower()

    return {"dataset": cls.replace("_test", "")}


def get_features(opt):
    cls = opt["features__cls"]

    pat = re.compile(r"^<class '.*?\.feeds\.(.*?)'>$")
    cls = pat.sub(r"\1", cls).lower()

    if cls == "feedfourierfeatures":
        features = "fft"

    elif cls == "feedrawfeatures":
        features = "raw"
    else:
        raise ValueError("Unknown input features.")

    return {"features": features}
