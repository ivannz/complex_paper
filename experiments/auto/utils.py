import re
import time
import gzip

import torch
import importlib

from torch.utils.data import DataLoader


def feed_mover(feed, **kwargs):
    """Transfer to device and type cast the batches from the feed on the fly.

    Parameters
    ----------
    feed : torch.utils.data.DataLoader
        The data loader instance draw batches from.

    **kwargs : keyword arguments
        The keyword arguments for the method `torch.Tensor.to()`.

    Yields
    ------
    batch : iterable
        An iterable that consitutes the batch.
    """

    if not kwargs:
        yield from feed
        return

    for batch in feed:
        yield [b.to(**kwargs) for b in batch]


def feed_limiter(feed, max=-1):
    """Limit the number of batches requested from the feed.

    Parameters
    ----------
    feed : torch.utils.data.DataLoader
        The data loader instance draw batches from.

    max : int, default=-1
        The limit on the number of batches generated.
        Disabled if `max` is negative.

    Yields
    ------
    batch : iterable
        An iterable that consitutes the batch.
    """

    if max < 0:
        yield from feed
        return

    for batch, _ in zip(feed, range(max)):
        yield batch


def save_snapshot(filename, **kwargs):
    with gzip.open(filename, "wb", compresslevel=5) as fout:
        torch.save(dict(kwargs, **{
            "__timestamp__": time.strftime("%Y%m%d-%H%M%S"),
        }), fout, pickle_protocol=3)

    return filename


def param_apply(param, **mapper):
    result = {}
    for name, value in param.items():
        if value is not None:
            fn = mapper.get(name)
            retvalue = fn(value) if callable(fn) else value
            value = value if retvalue is None else retvalue

        result[name] = value

    return result


def get_class(klass):
    """Parse the specified type-string, import it and return the type."""
    if isinstance(klass, type):
        return klass

    if not isinstance(klass, str):
        return None

    # match and rsplit by "."
    match = re.fullmatch(r"^<class\s+'(?:(.*)\.)?([^\.]+)'>$", klass)
    if match is None:
        return None

    # import from builtins if no module is specified
    module, klass = (match.group(1) or "builtins"), match.group(2)
    return getattr(importlib.import_module(module), klass)


def get_instance(*args, cls, **options):
    """Locate and create a `cls` instance."""
    return get_class(cls)(*args, **options)
