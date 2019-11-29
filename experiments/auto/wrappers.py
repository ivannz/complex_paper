import torch
from torch.utils.data import DataLoader


class FeedWrapper(object):
    """A wrapper for a dataLoader that puts batches on device on the fly.

    Parameters
    ----------
    feed : torch.utils.data.DataLoader
        The data loader instance to be wrapped.

    **kwargs : keyword arguments
        The keyword arguments to be passed to `torch.Tensor.to()`.
    """
    def __init__(self, feed, **kwargs):
        assert isinstance(feed, DataLoader)
        self.feed, self.kwargs = feed, kwargs

    def __len__(self):
        return len(self.feed)

    def __iter__(self):
        if not self.kwargs:
            yield from iter(self.feed)

        else:
            for batch in iter(self.feed):
                yield tuple(b.to(**self.kwargs)
                            for b in batch)
