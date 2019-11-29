import re
import time
import gzip

import torch
import importlib


def recipe_param(recipe, **defaults):
    param = {}
    if recipe and isinstance(recipe[-1], dict):
        *recipe, param = recipe

    return recipe, {**defaults, **param}


def recipe_param_apply(recipe, **mapper):
    def apply(param):
        return {
            name: mapper.get(name, lambda v: v)(value) or value
            for name, value in param.items()
        }

    def walk(recipe):
        if isinstance(recipe, (tuple, list)) and recipe:
            param = {}
            if isinstance(recipe[-1], dict):
                *recipe, param = recipe

            return (*map(walk, recipe), apply(param))
        return recipe

    return walk(recipe)


def get_class(klass):
    if not isinstance(klass, str):
        return None

    # match and rsplit by "."
    match = re.fullmatch(r"^<class\s+'(?:(.*)\.)?([^\.]+)'>$", klass)
    if match is None:
        return None

    # import from builtins if no module is specified
    module, klass = (match.group(1) or "builtins"), match.group(2)
    return getattr(importlib.import_module(module), klass)


def get_instance(*args, **options):
    factory = get_class(options.pop("class"))
    return factory(*args, **options)


def get_model_factory(**options):
    from functools import partial
    from .. import models

    options.pop("repr", None)

    klass = options.pop("class")
    factory = get_class(klass) or getattr(models, klass)

    recipe = recipe_param_apply(
        options.pop("recipe", []), activation=get_class)
    return partial(factory, *recipe, **options)
