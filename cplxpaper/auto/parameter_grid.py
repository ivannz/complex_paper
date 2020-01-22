"""Service code pertaining to manipulating of parameters for manifests."""
from collections import defaultdict


def get_params(parameters, deep=True, keepcontainers=True):
    """Depth first redundantly flatten a nested dictionary.

    Arguments
    ---------
    parameters : dict
        The dictionary to traverse and linearize.

    deep : bool, default=True
        Whether to perform depth first traversal of nested dictionaries
        or not.

    keepcontainers : bool, default=True
        Whether to keep return the nested containers (dicts) or not.
        Effective only if `deep` is `True`.

    Details
    -------
    Adapted from scikit's BaseEstimator. Does not handle recursive
    dictionaries.
    """
    out = {}
    for key in parameters:
        value = parameters[key]
        if deep and isinstance(value, dict):
            nested = get_params(value, deep=True, keepcontainers=keepcontainers)
            out.update((key + '__' + k, val) for k, val in nested.items())
            if not keepcontainers:
                continue

        out[key] = value

    return out


def set_params(parameters, **params):
    """In-place update of a nested dictionary.

    Details
    -------
    Adapted from scikit's BaseEstimator. Does not handle recursive
    dictionaries.
    """
    if not params:
        return parameters

    nested_params = defaultdict(dict)
    for key, value in params.items():
        key, delim, sub_key = key.partition('__')
        if delim:
            nested_params[key][sub_key] = value

        else:
            parameters[key] = value

    for key, sub_params in nested_params.items():
        set_params(parameters[key], **sub_params)

    return parameters


def special_params(**parameters):
    """Returns a pair (parameters, special).

    Details
    -------
    Special parameters are those key that begin with '__'.
    """
    special = set(k for k in parameters if k.startswith("__"))
    return {k: v for k, v in parameters.items() if k not in special}, \
           {k[2:]: v for k, v in parameters.items() if k in special}


def flatten(parameters):
    """Depth first flatten a nested dictionary and .

    Arguments
    ---------
    parameters : dict
        The dictionary to traverse and linearize.

    Details
    -------
    See details in `get_params`.
    """
    flat = get_params(parameters, deep=True, keepcontainers=False)

    out = {}
    for key in flat:
        value = flat[key]
        if not isinstance(value, (list, tuple)):
            out[key] = value
            continue

        # convert tuples and lists to dict
        nested = flatten({f"[{i}]": v for i, v in enumerate(value)})
        out.update((key + k, val) for k, val in nested.items())

    return out
