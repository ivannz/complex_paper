"""Measure performance of the models.

This module is exceptionally task-specific, except for, probabily,
`predict()` and `ExtremeTracker`,
"""
import math
import numpy as np
import torch
import warnings

from scipy.special import expit as sigmoid

from sklearn.metrics import confusion_matrix as base_confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve

from cplxmodule.utils.stats import sparsity

from ..auto.feeds import feed_forward_pass

from ..auto.performance import BaseEarlyStopper, BasePerformanceEvaluation


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


class MusicNetBasePerformance(BasePerformanceEvaluation):
    def __init__(self, feed, curves=False, threshold=-0.5):
        super().__init__(feed)
        self.curves, self.threshold = curves, threshold

    @classmethod
    def eval_impl(cls, model, feed, curves=False, threshold=-0.5):
        """Compute the multi-output binary classification performance metrics."""
        out = {
            "sparsity": sparsity(model, threshold=threshold, hard=True)
        }

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

        # Raw per output average precision
        out["average_precision"] = average_precision_score(
            y_true, logits, pos_label=1, average=None)

        # Pooled AP (treating different outputs as one -- good?) -- slow
        out["pooled_average_precision"] = average_precision_score(
            y_true.ravel(), logits.ravel(), pos_label=1, average=None)

        # compute the curves
        out["ap_curves"] = {}
        if curves:
            y_prob = sigmoid(logits)
            out["ap_curves"]["pooled"] = precision_recall_curve(
                y_true.ravel(), y_prob.ravel())

            out["ap_curves"].update({
                j: precision_recall_curve(y_true[:, j], y_prob[:, j])
                for j in range(n_outputs)})
            # curves take too much space 2 x (3 x len x float64)!

        return out


class PooledAveragePrecisionEarlyStopper(BaseEarlyStopper):
    def __init__(self, scorer, cooldown=1, patience=10, rtol=1e-3, atol=1e-4,
                 raises=StopIteration):
        super().__init__(extreme="max", cooldown=cooldown, patience=patience,
                         rtol=rtol, atol=atol, raises=raises)
        self.scorer = scorer

    def get_score(self, model):
        # evaluate the `model`, toggles eval mode
        scores = self.scorer(model.eval())
        return scores["pooled_average_precision"]
