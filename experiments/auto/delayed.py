import signal

from inspect import ismethod


class DelayedKeyboardInterrupt(object):
    """Create an atomic section with respect to the Keyboard Interrupt.

    Parameters
    ----------
    action : str, ("delay", "raise", "ignore")
        The desired behaviour on KeyboardInterrupt. If action is "raise"
        then this effectively disables the critical section and passes
        the keyboard interrupt events through. If action is "delay", then
        the events are delayed until the scope of the critical section is
        left. If action is "ignore" then the events are captured, but
        ignored.

    Details
    -------
    Typically used in the `with` statement in the following manner:
    >>> with DelayedKeyboardInterrupt("ignore") as flag:
    >>>     for i in range():
    >>>         ...
    >>>         if flag:
    >>>             break
    """
    def __init__(self, action="delay"):
        assert action in ("raise", "delay", "ignore")
        self.action = action

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.action)})"

    def __bool__(self):
        return self.signal is not None

    def __enter__(self):
        self.signal, self.is_nested_ = None, False

        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

        if ismethod(self.old_handler):
            self.is_nested_ = isinstance(self.old_handler.__self__,
                                         DelayedKeyboardInterrupt)

        return self

    def handler(self, sig, frame):
        self.signal = sig, frame

        if self.action == "raise" or self.is_nested_:
            self.old_handler(sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.action == "delay" and self.signal:
            self.old_handler(*self.signal)
