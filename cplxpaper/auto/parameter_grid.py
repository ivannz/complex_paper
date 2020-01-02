from collections import defaultdict


def get_params(self, deep=True, keepcontainers=True):
    """Depth first redundantly flatten a nested dictionary.

    Arguments
    ---------
    self : dict
        The dictionary to traverser and linearize.

    deep : bool, default=True
        Whether to perform depth first travseral of nested dictionaries
        or not.

    keepcontainers : bool, default=True
        Whether to keep return the nested containers (dicts) or not.
        Effective only if `deep` is `True`.
    """
    out = dict()
    for key in self:
        value = self[key]
        if deep and isinstance(value, dict):
            nested = get_params(value, deep=True, keepcontainers=keepcontainers)
            out.update((key + '__' + k, val) for k, val in nested.items())
            if not keepcontainers:
                continue

        out[key] = value

    return out


def set_params(self, **params):
    """Inplace update of a nested dictionary.

    Details
    -------
    Adapted from scikit's BaseEstimator. Does not handle
    recurusive dictionaries.
    """
    if not params:
        return self

    nested_params = defaultdict(dict)
    for key, value in params.items():
        key, delim, sub_key = key.partition('__')
        if delim:
            nested_params[key][sub_key] = value

        else:
            self[key] = value

    for key, sub_params in nested_params.items():
        set_params(self[key], **sub_params)

    return self


def special_params(**params):
    """Returns a pair (params, special).

    Details
    -------
    Special parameters are those key that begin with '__'.
    """
    special = set(k for k in params if k.startswith("__"))
    return {k: v for k, v in params.items() if k not in special}, \
           {k[2:]: v for k, v in params.items() if k in special}
