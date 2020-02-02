import os
import re
import time
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

    return state


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


def collate_history(history, fields):
    """Repack the list of tuples into a dict of arrays keyed by `fields`.

    Examples
    --------
    >>> collate_history([(1., 2.), (3., 4.)], ["a", "b"])
    {'a': array([1., 3.]), 'b': array([2., 4.])}
    """
    if history:
        values = map(np.array, zip(*history))

    else:
        values = map(np.empty, repeat(0, len(fields)))

    return dict(zip(fields, values))


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

    def decorator(user_function):
        def wrapper(*args, **kwargs):
            # dumb stategy: pickle the args and use it as a binary key
            cache, key = {}, pickle.dumps((args, kwargs))
            if isinstance(cachename, str) and os.path.exists(cachename):
                with open(cachename, "rb") as fin:
                    cache = pickle.load(fin)

            if key not in cache or recompute:
                cache[key] = user_function(*args, **kwargs)

            if isinstance(cachename, str):
                with open(cachename, "wb") as fout:
                    pickle.dump(cache, fout)

            return cache[key]

        return update_wrapper(wrapper, user_function)

    return decorator
