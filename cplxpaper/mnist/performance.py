import torch
import numpy as np

from sklearn.metrics import confusion_matrix
from cplxmodule.utils.stats import named_sparsity

from ..auto.performance import BasePerformanceEvaluation, BaseEarlyStopper
from ..auto.feeds import feed_forward_pass


def predict(model, feed):
    """Collect the logit scores, true and predicted labels from the feed."""
    feed_pred = feed_forward_pass(feed, model)
    logits, y_true = map(np.concatenate, zip(*feed_pred))

    return y_true, logits.argmax(-1), logits


class MNISTBasePerformance(BasePerformanceEvaluation):
    def __init__(self, feed, threshold=-0.5):
        super().__init__(feed)
        self.threshold = threshold

    @classmethod
    def eval_impl(cls, model, feed, threshold):
        """Compute the multiclass performance metrics."""
        out = {
            "sparsity": dict(named_sparsity(model, threshold=threshold, hard=True))
        }

        model.eval()
        y_true, y_pred, logits = predict(model, feed)

        cm = confusion_matrix(y_true, y_pred)
        tp = cm.diagonal()
        fp, fn = cm.sum(axis=1) - tp, cm.sum(axis=0) - tp

        out["accuracy"] = tp.sum() / cm.sum()           # ~ P(\hat{y} = y)
        out["precision"] = tp / np.maximum(tp + fp, 1)  # ~ P(y=1 \mid \hat{y}=1)
        out["recall"] = tp / np.maximum(tp + fn, 1)     # ~ P(\hat{y}=1 \mid y=1)

        return out


class AccuracyEarlyStopper(BaseEarlyStopper):
    def __init__(self, scorer, cooldown=1, patience=10, rtol=1e-3, atol=1e-4,
                 raises=StopIteration):
        super().__init__(extreme="max", cooldown=cooldown, patience=patience,
                         rtol=rtol, atol=atol, raises=raises)
        self.scorer = scorer

    def get_score(self, model):
        # evaluate the `model`, toggles eval mode
        scores = self.scorer(model.eval())
        return scores["accuracy"]
