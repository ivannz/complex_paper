import torch


class BaseFeedWrapper(object):
    """A base class for feed wrappers."""
    def __init__(self, feed):
        self.feed = feed

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.feed)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.feed)})"


class FeedMover(BaseFeedWrapper):
    def __init__(self, feed, **devtype):
        super().__init__(feed)
        self.devtype = devtype

    def __iter__(self):
        return self.iter_impl(self.feed, **self.devtype)

    @classmethod
    def iter_impl(cls, feed, **devtype):
        """Transfer to device and type cast the batches from the feed on the fly.

        Parameters
        ----------
        feed : iterable
            The data loader instance to draw batches from.

        **devtype : keyword arguments
            The keyword arguments (device, dtype, ...) for the method
            `torch.Tensor.to()`.

        Yields
        ------
        batch : iterable
            An iterable that constitutes the batch.
        """
        if devtype:
            for batch in feed:
                yield [b.to(**devtype) for b in batch]

        else:
            yield from feed


class FeedLimiter(BaseFeedWrapper):
    def __init__(self, feed, max_iter=-1):
        super().__init__(feed)
        self.max_iter = max_iter

    def __iter__(self):
        return self.iter_impl(self.feed, self.max_iter)

    @classmethod
    def iter_impl(cls, feed, max=-1):
        """Limit the number of batches requested from the feed.

        Parameters
        ----------
        feed : iterable
            The data loader instance to draw batches from.

        max : int, default=-1
            The limit on the number of batches generated.
            Disabled if `max` is negative.

        Yields
        ------
        batch : iterable
            An iterable that constitutes the batch.
        """

        if max >= 0:
            for batch, _ in zip(feed, range(max)):
                yield batch

        else:
            yield from feed

    def __len__(self):
        if self.max_iter < 0:
            return len(self.feed)
        return min(len(self.feed), self.max_iter)


def feed_forward_pass(feed, module):
    """Feed the first item in the batch tuple through the provided module.

    Parameters
    ----------
    feed : iterable
        The data loader instance to draw batches from.

    module : torch.nn.Module
        The module which the first element in the batch is fed to.

    Yields
    ------
    batch : iterable
        An iterable that constitutes the batch.

    Details
    -------
    Does not force the module into eval mode! Disables gradients, computes a
    forward pass moves the resulting batch to CPU and converts to numpy.
    """
    def to_numpy(t):  # assumes detached tensor or no_grad
        return t.cpu().numpy()

    with torch.no_grad():
        for X, *rest in feed:
            yield [*map(to_numpy, (module(X), *rest))]
