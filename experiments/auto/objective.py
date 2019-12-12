"""Handy weighted composite objectives and penalties."""
import ast
import operator

import torch

from torch.nn.modules import Module
from cplxmodule.relevance.base import BaseARD


def named_ard_modules(module, prefix=""):
    """Generator of named variational dropout modules."""
    for name, submod in module.named_modules(prefix=prefix):
        if isinstance(submod, BaseARD):
            yield name, submod


class BaseObjective(Module):
    """Base class for objective functions.

    Parameters
    ----------
    reduction : str, default="mean"
        The reduction method, for compatibility with Loss.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction == "mean":
            self.reduce_fn = torch.mean

        elif reduction == "sum":
            self.reduce_fn = torch.sum

        else:
            raise ValueError(f"`reduction` must be 'mean' or "
                             f"'sum'. Got '{reduction}'.")

        self.reduction = reduction

    def forward(self, module, input=None, target=None, output=None):
        """Forward prototype.

        Keyword Arguments
        -----------------
        module : torch.nn.Module
            The module for which to compute the penalty, in case special
            layers and hooks are used.

        input : torch.Tensor, or tuple thereof
            The input.

        target : torch.Tensor, or tuple thereof
            The targets.

        output : torch.Tensor, or tuple thereof
            The precomputed forward pass of through the model.
        """
        raise NotImplementedError


class ARDPenaltyObjective(BaseObjective):
    """Penalty for variational dropout modules.

    Parameters
    ----------
    coef : dict, callable, or float
        The coefficients for the variational submodules. A dictionary
        keeps the constant weight for each variational submodule, a
        constant specifies a uniform coefficient for all penalties,
        and a callable specifies a dynamically computed coefficient.

    reduction : str, default="mean"
        The reduction method, for compatibility with Loss.
    """

    def __init__(self, coef=1., reduction="sum"):
        super().__init__(reduction=reduction)
        if isinstance(coef, dict):
            coef = coef.get
        self.coef = coef if callable(coef) else lambda n: coef

    def forward(self, module, input=None, target=None, output=None):
        """Reimplements `named_penalties` with non-uniform coefficients."""
        # get names of variational modules and pre-fetch coefficients
        ardmods = dict(named_ard_modules(module))
        coef_fn, fn = self.coef, self.reduce_fn
        if isinstance(coef_fn, (int, float)) and coef_fn > 0:
            return coef_fn * sum(fn(mod.penalty) for mod in ardmods.values())

        # lazy-compute the weighted sum
        coefs = [1. if C is None else C for C in map(coef_fn, ardmods)]
        return sum(C * fn(m.penalty) for C, m in zip(coefs, ardmods.values()) if C > 0)


class BaseCompositeObjective(BaseObjective):
    """Differentiable objective defined as a weighted sum.

    Parameters
    ----------
    **terms : keyword arguments
        A mapping of the variable names to objective terms.
    """

    def __init__(self, **terms):
        from torch.nn.modules.loss import _Loss  # import locally
        assert all(isinstance(term, (_Loss, BaseObjective))
                   for term in terms.values())

        super().__init__(reduction="sum")

        self.terms = torch.nn.ModuleDict(terms)

    def extra_repr(self):
        return repr(list(self.terms))

    def __getitem__(self, key):
        return self.terms[key]

    def __dir__(self):
        return list(self.terms)

    def forward(self, module, input, target, output=None):
        """Evaluate the terms of the objective.

        Parameters
        ----------
        module : torch.nn.Module
            The module, which to evaluate with. Passed as is to `BaseObjective`
            subclasses, so that they can access meta information within the
            module.

        input : torch.Tensor, or tuple thereof
            The inputs corresponding to the target.

        target : torch.Tensor, or tuple thereof
            The target outputs of the model.

        Details
        -------
        Precomputes the output of the module on the given input.
        """

        # evaluate the components and cache the most recent values
        components = {}
        output = module(input) if output is None else output
        for name, term in self.terms.items():
            if isinstance(term, BaseObjective):
                components[name] = term(module, input, target, output)

            else:
                components[name] = term(output, target)

        self.component_values_ = tuple(map(float, components.values()))
        return components


class WeightedObjective(BaseCompositeObjective):
    """Differentiable objective defined as a weighted sum.

    Parameters
    ----------
    weights : mapping
        A mapping from variable names to weight values.

    **terms : keyword arguments
        A mapping of the variable names to objective terms.
    """

    def __init__(self, weights, **terms):
        assert not (weights.keys() - terms.keys())
        super().__init__(**terms)

        self.weights = weights

    def extra_repr(self):
        return repr(self.weights)

    def forward(self, module, input, target, output=None):
        components = super().forward(module, input, target, output)
        return sum(C * components[name] for name, C in self.weights.items())


class ScalarExpression(ast.NodeVisitor):
    """Evaluate a simple arithmetic expression within a namespace."""
    operators = {
        ast.Add:  operator.add,
        ast.UAdd: operator.pos,
        ast.Sub:  operator.sub,
        ast.USub: operator.neg,
        ast.Mult: operator.mul,
        ast.Div:  operator.truediv,
    }

    @classmethod
    def validate(cls, expression, **namespace):
        """Validate a scalar expression with specified variables.

        Details
        -------
        All variables in the namespace assume a test value of float(1),
        and then the expression is parsed and its syntax tree validated
        for non-arithmetic operations, e.g. indexing or calling, and
        non-variables, like containers or other rich objects.
        """
        try:
            visitor = cls(**dict.fromkeys(namespace, 1.))
            visitor.visit(ast.parse(expression, mode="eval").body)

        except SyntaxError:
            raise RuntimeError(f"Bad expression `{expression}`") from None

        except KeyError as e:
            msg = f"{repr(e.args)} not found among {repr(list(namespace))}"
            raise RuntimeError(msg) from None

    def __init__(self, **namespace):
        self.namespace = namespace

    def generic_visit(self, node):
        """Raise a SyntaxError on unhandled nodes."""
        raise SyntaxError()

    def resolve_op(self, node):
        try:
            return self.operators[type(node.op)]

        except KeyError:
            raise SyntaxError() from None

    def visit_UnaryOp(self, node):
        """Evaluate a unary +,- on the node's child."""
        op = self.resolve_op(node)
        return op(self.visit(node.operand))

    def visit_BinOp(self, node):
        """Evaluate a binary +, -, *, / on the node's children."""
        op = self.resolve_op(node)
        return op(self.visit(node.left), self.visit(node.right))

    def visit_Num(self, node):
        """Evaluate a numeric constant, i.e. int, float, or complex."""
        return node.n

    def visit_Name(self, node):
        """Fetch a variable from the namespace."""
        return self.namespace[node.id]


class ExpressionObjective(BaseCompositeObjective):
    """Differentiable objective defined though a valid arithmetic expression.

    Parameters
    ----------
    expression : str
        A valid pythonic expression that involves variables, specified in
        terms. Allowed to include arithmetic binary and unary operators,
        numeric constants and syntactically valid variable names.

    **terms : keyword arguments
        A mapping of the variable names to objective terms.
    """

    def __init__(self, expression, **terms):
        ScalarExpression.validate(expression, **terms)
        super().__init__(**terms)

        self.expression = expression
        self.compiled = compile(expression, str(self), mode="eval", optimize=2)

    def extra_repr(self):
        return repr(self.expression)

    def forward(self, module, input, target, output=None):
        """Evaluate a safe arithmetic expression with the components' values"""
        components = super().forward(module, input, target, output)
        return eval(self.compiled, components)
