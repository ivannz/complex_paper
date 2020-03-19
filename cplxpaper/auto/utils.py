import os
import re
import time
import json
import gzip
import pickle

import numpy as np

import torch
import importlib

from functools import partial, update_wrapper
from itertools import repeat


def save_snapshot(filename, **kwargs):
    with gzip.open(filename, "wb", compresslevel=5) as fout:
        torch.save(dict(kwargs, **{
            "__timestamp__": time.strftime("%Y%m%d-%H%M%S"),
        }), fout, pickle_protocol=3)

    return filename


def load_snapshot(filename):
    with gzip.open(filename, "rb") as fin:
        state = torch.load(fin, map_location=torch.device("cpu"))

    # verify version
    pass

    return state


def load_manifest(folder, create=True):
    """Load the manifest of an experiment, falling back to snapshots if missing."""
    # check if `config.json` resides at 'folder'
    manifest = os.path.join(folder, "config.json")
    if os.path.exists(manifest):
        with open(manifest, "r") as fin:
            return json.load(fin)

    # synchronized with naming format at ./auto.py#L393
    pat = re.compile(r"^\d+-.*\.gz$")

    _, _, filenames = next(os.walk(folder))
    for match in filter(None, map(pat.match, filenames)):
        # load the first snapshot and quit
        snapshot = load_snapshot(os.path.join(folder, match.string))
        if create:
            with open(manifest, "w") as fout:
                json.dump(snapshot["options"], fout, indent=2, sort_keys=False)

        return snapshot["options"]

    raise RuntimeError(f"Could not load config at '{folder}'")


def verify_experiment(folder):
    """Check if the experiment at the specified folder has been completed."""
    if not os.path.isdir(folder):
        return False

    filename = os.path.join(folder, "config.json")
    if not os.path.isfile(filename):
        return False

    with open(filename, "r") as fin:
        manifest = json.load(fin)

    # stage map to check if any one is missing
    folder, _, filenames = next(os.walk(folder))
    stages = dict.fromkeys(manifest["stage-order"], False)
    for j, stage in enumerate(stages.keys()):
        # synchronized with naming format at ./auto.py#L393
        pat = re.compile(f"^{j}-{stage}\\s+.*\\.gz$")
        match = next(filter(None, map(pat.match, filenames)), None)
        stages[stage] = match is not None

    return all(stages.values())


def get_stage_snapshot(stage, folder):
    """Get the filename of the snapshot of the given phase from the experiment.
    """
    # synchronized with naming format at ./auto.py#L393
    pat = re.compile(r"^(\d+)-(\S+)\s+.*?(\d{8}-\d{6})\.gz$")

    # list all filenames matching the format
    folder, _, filenames = next(os.walk(folder))
    matches = filter(None, map(pat.match, filenames))

    # search from the most recently modified to the oldest snapshot
    matches = sorted(matches, reverse=True,
                     key=lambda m: time.strptime(m.group(3), "%Y%m%d-%H%M%S"))
    for m in matches:
        if m.group(2) == stage:
            return os.path.join(folder, m.string)

    raise RuntimeError(f"'{stage}' not found.")


def load_stage_snapshot(stage, folder, mismatch="ignore"):
    """Load the snapshot of the given phase from the experiment."""
    if mismatch not in ("ignore", "raise"):
        raise ValueError(f"`mismatch` must be either 'ignore' or 'raise'.")

    try:
        # load and validate stage ID
        snapshot = load_snapshot(get_stage_snapshot(stage, folder))

        # the snapshot is expected to comply with ./auto.py#L392-411
        if snapshot["stage"][0] == stage:
            return snapshot

    except RuntimeError:
        if mismatch != "ignore":
            raise

    return {}


def param_defaults(param, **defaults):
    """Add keys with default values if absent.

    Parameters
    ----------
    param : dict
        The dictionary of parameters.

    **defaults : keyword arguments
        The default values of the missing keys.

    Returns
    -------
    out : dict
        A shallow copy of the parameters with specified (shallow) defaults.
    """
    return {**defaults, **param}


def param_apply_map(param, deep=True, memo=None, **map):
    """Recursively re-map the values within the nested dictionary.

    Parameters
    ----------
    param : dict
        The dictionary of parameters.

    deep : bool, default=True
        Whether to recursively copy the nested dictionaries.

    memo : dict, default=None
        Internally used service dictionary to keep track of visited nested
        dictionaries.

    **map : keyworded functions
        The key-function mapping, applied to non-dictionary elements.

    Returns
    -------
    out : dict
        A shallow or deep modified copy of the original parameter dictionary.

    Details
    -------
    If `map` is provided, then produces a modified copy of the original
    dictionary, maintaining is key ordering.
    """
    if not map:
        return param

    # keep track of visited nested dictionaries by their `id(...)`
    if memo is None:
        memo = {}

    # nested self reference: return the previously computed dictionary
    if id(param) in memo:
        return memo[id(param)]

    out = {}
    memo[id(param)] = out  # reference the dynamically created dictionary

    # make an elementwise copy of the dictionary
    for key in param:
        value = param[key]
        # prioritize dfs, then pass the nested result through the map
        if isinstance(value, dict) and deep:
            value = param_apply_map(value, deep=True, memo=memo, **map)

        out[key] = value

    # map all non None values using the supplied map
    for key in out:
        value = out[key]
        if key in map and value is not None:
            out[key] = map[key](value)

    return out


def get_class(klass):
    """Parse the specified type-string, import it and return the type."""
    if isinstance(klass, type):
        return klass

    if not isinstance(klass, str):
        raise TypeError(f"Expected a string, got {type(klass)}.")

    # match and rsplit by "."
    match = re.fullmatch(r"^<class\s+'(?:(.*)\.)?([^\.]+)'>$", klass)
    if match is None:
        raise ValueError(f"{klass} is not a type identifier.")

    # import from built-ins if no module is specified
    module, klass = (match.group(1) or "builtins"), match.group(2)
    return getattr(importlib.import_module(module), klass)


def get_instance(*args, cls, **options):
    """Locate and create a `cls` instance."""
    return get_class(cls)(*args, **options)


def get_factory(*args, cls, **options):
    """Locate and create a partial `cls` constructor."""
    return partial(get_class(cls), *args, **options)


def join(*, left, right, how="inner", f=None):
    """SQL-like join of two dictionaries."""
    assert isinstance(left, dict) and isinstance(right, dict)

    if how == "left":
        key = left.keys()

    elif how == "right":
        key = right.keys()

    elif how == "inner":
        key = left.keys() & right.keys()

    else:  # how == "outer"
        key = left.keys() | right.keys()

    def f_or_none(x):
        return None if x is None else (x if not callable(f) else f(x))

    return {k: (f_or_none(left.get(k)), f_or_none(right.get(k))) for k in key}


def deploy_optimizer(mapper, *, target, source=None):
    """Copy common state from the `source` to the `target` optimizer.

    Note
    ----
    Neither learning rate settings nor parameter groups are copied.
    """
    if source is None:
        return target

    if isinstance(source, torch.optim.Optimizer):
        d_source = source.state_dict()

    elif isinstance(source, dict):
        d_source = source

    else:
        raise TypeError(f"`source` must be a dict or an optimizer. "
                        f"Got {type(source).__name__}.")

    d_target = target.state_dict()

    # transfer the state
    d_target["state"].update({
        mapper[k]: v for k, v in d_source["state"].items() if k in mapper
    })

    target.load_state_dict(d_target)
    return target


def state_dict_to_cpu(state_dict):
    """Make a copy of the state dict onto the cpu."""
    # .state_dict() references tensors, so we detach and copy to cpu
    return {key: par.detach().cpu() for key, par in state_dict.items()}


class TermsHistory(object):
    def __init__(self, *fields):
        self.records = {k: [] for k in fields}

    def append(self, record):
        for k, v in record.items():
            self.records.setdefault(k, []).append(v)

    def __getitem__(self, key):
        return np.array(self.records[key])

    def __iter__(self):
        yield from ((k, np.array(v)) for k, v in self.records.items())


def filter_prefix(dictionary, *prefixes):
    """Return dict with keys having any of the prefixes."""
    if not prefixes:
        return dictionary

    out = {}
    for key, value in dictionary.items():
        if any(map(str(key).startswith, prefixes)):
            out[key] = value

    # may be empty
    return out


def file_cache(cachename, recompute=False):
    """File-based cache decorator.

    Parameters
    ----------
    cachename : str, or None
        The name of the file to use for cache storage. Caching
        is disabled if this is set to `None`.

    recompute : bool, default=False
        Whether to update the cache on every call.

    Details
    -------
    Arguments (keyworded and positional) to the cached function must
    be hashable, because they are used for lookup. In essence, the
    arguments are serialized and the resulting binary string is used
    as key.
    """
    assert cachename is None or isinstance(cachename, str)

    if not isinstance(cachename, str):
        # no filename given : decorator just passes function through
        def decorator(user_function):
            return user_function

        return decorator

    # initialize an empty cache
    if not os.path.exists(cachename):
        with open(cachename, "wb") as fout:
            pickle.dump({}, fout)

    def decorator(user_function):
        with open(cachename, "rb") as fin:
            cache = pickle.load(fin)

        def wrapper(*args, **kwargs):
            # dumb stategy: pickle the args and use it as a binary key
            key = pickle.dumps((args, kwargs), protocol=3)

            # no key: call the wrapped function
            if key not in cache or recompute:
                cache[key] = user_function(*args, **kwargs)

                with open(cachename, "wb") as fout:
                    pickle.dump(cache, fout)

            return cache[key]

        return update_wrapper(wrapper, user_function)

    return decorator
