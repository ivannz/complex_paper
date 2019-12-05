"""Measure performance of the models.

This module is exceptionally task-specific, except for, probabily,
`predict()` and `ValueTracker`,
"""
import math
import numpy as np

from scipy.special import expit as sigmoid

from sklearn.metrics import confusion_matrix as base_confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from .utils import feed_forward_pass


def confusion_matrix(y_true, y_pred, fast=True):
    r"""Compute the confusion matrix for a binary classification problem.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    fast : bool, default=True
        Whether to use faster counting or not.

    Returns
    -------
    confusion matrix: array, shape=(2, 2)
        The confusion matrix C_{ij} of a binary classification task:
        C_{ij} -- # y_k = i & \hat{y}_k = j, e.g. C_{10} is false negative.

    Details
    -------
    Very specialized, so faster than sklearn's reference function. Works only
    if the inputs guaranteed to be binary, i.e. {0, 1}, but doesn't validate
    the inputs. Encodes predicted and true labels with `pred + true * 2`.
    """
    if fast:
        # Assumes binary array input (not validated!)
        #  array([tn, fp, fn, tp]).reshape(2, 2)
        return np.bincount(y_pred + y_true * 2, minlength=4).reshape(2, 2)

    # uses multiprocessing
    return base_confusion_matrix(y_true, y_pred, labels=[0, 1])


def predict(model, feed):
    """Collect the logit scores, true and predicted labels from the feed."""
    feed_pred = feed_forward_pass(feed, model)
    logits, y_true = map(np.concatenate, zip(*feed_pred))

    y_true, y_pred = y_true.astype(np.int8), (logits >= 0).astype(np.int8)
    return y_true, y_pred, logits


def evaluate(model, feed, curves=False):
    """Compute the multi-output binary classification performance metrics."""
    out = {}

    model.eval()
    y_true, y_pred, logits = predict(model, feed)

    # get the binary classification metrics for each output
    n_samples, n_outputs = y_true.shape
    (tn, fp), (fn, tp) = np.stack([
        confusion_matrix(y_true[:, j], y_pred[:, j])
        for j in range(n_outputs)
    ], axis=-1)

    # Gotta compute'em all by hand!
    out["accuracy"] = (tp + tn) / (tp + tn + fp + fn)  # ~ P(\hat{y} = y)
    out["precision"] = tp / np.maximum(tp + fp, 1)  # ~ P(y=1 \mid \hat{y}=1)
    out["recall"] = tp / np.maximum(tp + fn, 1)     # ~ P(\hat{y}=1 \mid y=1)

    # Raw per output average precision (actually logits should suffice)
    y_prob = sigmoid(logits)
    out["average_precision"] = average_precision_score(
        y_true, y_prob, pos_label=1, average=None)

    # Pooled AP (treating different outputs as one -- good?) -- slow
    out["pooled_average_precision"] = average_precision_score(
        y_true.ravel(), y_prob.ravel(), pos_label=1, average=None)

    # compute the curves
    out["ap_curves"] = {}
    if curves:
        out["ap_curves"]["pooled"] = precision_recall_curve(
            y_true.ravel(), y_prob.ravel())

        out["ap_curves"].update({
            j: precision_recall_curve(y_true[:, j], y_prob[:, j])
            for j in range(n_outputs)})
        # curves take too much space 2 x (3 x len x float64)!

    return out


class ValueTracker(object):
    """Indicate if the metric stops improving for several epochs in a row.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs with no significant improvement in the tracked value
        after which early stopping should mechanics kick in.

    mode : str, default='min'
        In mode='min' the quantity is monitored for significant decreases,
        otherwise it is tracked for increases.

    rtol : float, default=1e-4
        The maximum allowed difference between the tracked value and its
        historical best, relative to the latter to consider a change
        insignificant.

    atol : float, default=0.
        The maximum absolute difference to consider a change insignificant.
    """
    def __init__(self, patience=10, mode="min", rtol=1e-4, atol=0.):
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"Unknown tracking mode `{mode}`")

        self.mode, self.rtol, self.atol = mode, rtol, atol
        self.patience = patience
        self.reset()

    def reset(self):
        self.best_ = -math.inf if self.mode == "max" else math.inf
        self.wait_, self.hits_ = 0, 0

    def is_worse(self, a, b):
        r"""Check if `a` is outside of the allowed tolerance of `b`."""
        delta = (a - b) if self.mode == "max" else (b - a)
        # (max) a \in (-\infty, b + |b| \rho + \alpha)
        # (min) a \in (b - |b| \rho - \alpha, +\infty)
        return delta < abs(b) * self.rtol + self.atol

    def step(self, value):
        """Decrease the time-to-live counter, depending on the metric."""
        current = float(value)
        if self.is_worse(current, self.best_):
            self.wait_ += 1

        else:
            self.best_, self.wait_, self.hits_ = current, 0, self.hits_ + 1

    def __bool__(self):
        return self.wait_ >= self.patience
