import math


class ExtremeTracker(object):
    """Indicate if the metric stops improving for several epochs in a row.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs with no significant improvement in the tracked value
        after which early stopping should mechanics kick in.

    extreme : str, default='min'
        In extreme='min' the quantity is monitored for significant decreases,
        otherwise it is tracked for increases.

    rtol : float, default=1e-4
        The maximum allowed difference between the tracked value and its
        historical best, relative to the latter to consider a change
        insignificant.

    atol : float, default=0.
        The maximum absolute difference to consider a change insignificant.
    """

    def __init__(self, patience=10, extreme="min", rtol=1e-4, atol=0.):
        super().__init__()
        if extreme not in ("min", "max"):
            raise ValueError(f"Unknown extreme `{extreme}`")

        self.extreme, self.rtol, self.atol = extreme, rtol, atol
        self.patience = patience
        self.reset()

    def reset(self):
        self.best_ = -math.inf if self.extreme == "max" else math.inf
        self.breached_, self.history_ = False, []
        self.wait_, self.hits_ = 0, 0

    def is_better(self, a, b, strict=True):
        r"""Check if `a` is within the allowed tolerance of `b` or `better`."""
        tau = 0 if strict else max(self.rtol * abs(b), self.atol)

        # (max) $a \in [b - \tau, +\infty]$, (min) $a \in [-\infty, b + \tau]$
        return (b - tau <= a) if self.extreme == "max" else (a <= b + tau)

    def step(self, value):
        """Decrease the time-to-live counter, depending on the provided value.

        Arguments
        ---------
        value : float
            The next observed value of the tracked metric.

        Returns
        -------
        reset : bool
            Flag, indicating if a new extreme value has been detected.
        """
        value = float(value)
        if self.is_better(value, self.best_, strict=True):
            self.wait_, self.breached_ = 0, False
            self.best_, self.hits_ = value, self.hits_ + 1

        elif self.is_better(value, self.best_, strict=False):
            self.wait_, self.breached_ = self.wait_ + 1, False

        else:
            self.wait_, self.breached_ = self.wait_ + 1, True

        self.history_.append(value)

        # indicate reset to the caller
        return self.wait_ == 0

    def __bool__(self):
        """Indicate if ran out of patience, or the last observed value
        breached the tracked band.
        """
        return self.wait_ >= self.patience or self.breached_

    @property
    def is_waiting(self):
        return not self


class BasePerformanceEvaluation(object):
    def __init__(self, feed):
        self.feed = feed

    def __call__(self, model):
        return self.eval_impl(model, **vars(self))

    @classmethod
    def eval_impl(cls, model, *, feed):
        raise NotImplementedError


class BaseEarlyStopper(ExtremeTracker):
    """Raise StopIteration if the metric stops improving for several epochs
    in a row.
    """

    def __init__(self, extreme, cooldown=1, patience=10,
                 rtol=1e-3, atol=1e-4, raises=StopIteration):
        assert raises is None or issubclass(raises, Exception)

        super().__init__(patience=patience, extreme=extreme,
                         rtol=rtol, atol=atol)
        self.cooldown, self.raises = cooldown, raises

    def reset(self):
        super().reset()
        self.last_epoch, self.next_epoch = -1, -math.inf

    def step(self, model, epoch=None):
        """Single step of the performance tracker.

        Arguments
        ---------
        epoch : int, default=None
            The current epoch number to override the internal epoch counter.

        Returns
        -------
        reset : bool
            Flag, indicating that a new extreme has been detected.
            See `:class:`~ExtremeTracker`.
        """
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.raises is not None and not self.is_waiting:
            raise self.raises

        # check if in cooldown mode and schedule next cooldown
        if self.last_epoch <= self.next_epoch:
            return

        self.next_epoch = self.last_epoch + self.cooldown

        return super().step(self.get_score(model))

    def get_score(self):
        raise NotImplementedError
