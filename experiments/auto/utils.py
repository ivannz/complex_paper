import re
import time
import gzip

import torch
import importlib

from functools import partial


class BaseFeedWrapper(object):
    """A base class for feed wrappers."""
    def __init__(self, feed):
        self.feed = feed

    def __repr__(self):
        return repr(self.feed)

    def __len__(self):
        return len(self.feed)

    def __iter__(self):
        raise NotImplementedError


def feed_mover(feed, **kwargs):
    """Transfer to device and type cast the batches from the feed on the fly.

    Parameters
    ----------
    feed : iterable
        The data loader instance to draw batches from.

    **kwargs : keyword arguments
        The keyword arguments for the method `torch.Tensor.to()`.

    Yields
    ------
    batch : iterable
        An iterable that constitutes the batch.
    """

    if not kwargs:
        yield from feed
        return

    for batch in feed:
        yield [b.to(**kwargs) for b in batch]


class FeedMover(BaseFeedWrapper):
    def __init__(self, feed, **kwargs):
        super().__init__(feed)
        self.devtype = kwargs

    def __iter__(self):
        yield from feed_mover(self.feed, **self.devtype)


def feed_limiter(feed, max=-1):
    """Limit the number of batches requested from the feed.

    Parameters
    ----------
    feed : iterable
        The data loader instance to draw batches from.

    max : int, default=-1
        The limit on the number of batches generated.
        Disabled if `max` is negative.

    Yields
    ------
    batch : iterable
        An iterable that constitutes the batch.
    """

    if max < 0:
        yield from feed
        return

    for batch, _ in zip(feed, range(max)):
        yield batch


class FeedLimiter(BaseFeedWrapper):
    def __init__(self, feed, max_iter=-1):
        super().__init__(feed)
        self.max_iter = max_iter

    def __len__(self):
        if self.max_iter < 0:
            return len(self.feed)
        return min(len(self.feed), self.max_iter)

    def __iter__(self):
        yield from feed_limiter(self.feed, self.max_iter)


def feed_forward_pass(feed, module):
    """Feed the first item in the batch tuple through the provided module.

    Parameters
    ----------
    feed : iterable
        The data loader instance to draw batches from.

    module : torch.nn.Module
        The module which the first element in the batch is fed to.

    Yields
    ------
    batch : iterable
        An iterable that constitutes the batch.

    Details
    -------
    Does not force the module into eval mode! Disables gradients, computes a
    forward pass moves the resulting batch to CPU and converts to numpy.
    """
    def to_numpy(t):  # assumes detached tensor or no_grad
        return t.cpu().numpy()

    with torch.no_grad():
        for X, *rest in feed:
            yield [*map(to_numpy, (module(X), *rest))]


def save_snapshot(filename, **kwargs):
    with gzip.open(filename, "wb", compresslevel=5) as fout:
        torch.save(dict(kwargs, **{
            "__timestamp__": time.strftime("%Y%m%d-%H%M%S"),
        }), fout, pickle_protocol=3)

    return filename


def load_snapshot(filename):
    with gzip.open(filename, "rb") as fin:
        state = torch.load(fin, map_location=torch.device("cpu"))

    # verify

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
