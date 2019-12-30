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


def torch_fftshift(tensor, dims=None):
    """Shift the zero-frequency component to the center of the spectrum.

    Details
    -------
    Taken almost verbatim from `scipy.fftpack.fftshift`."""
    if dims is None:
        dims = tuple(range(tensor.dim()))

    shape = tensor.shape
    if isinstance(dims, int):
        shift = shape[dims] // 2

    else:
        shift = [shape[ax] // 2 for ax in dims]

    return torch.roll(tensor, shift, dims)


def torch_rfft(input, signal_ndim, normalized=False):
    """Upcast the tensor to complex and compute fft.
    See `torch.fft` for details.
    """
    nil = torch.tensor(0.).to(input)

    input = torch.stack([input, nil.expand_as(input)], dim=-1)
    return torch.fft(input, signal_ndim, normalized)


def to_fourier(input, signal_ndim=1, complex=True, shift=False):
    """Compute the FFT of the real data.

    Parameters
    ----------
    input : tensor, shape = [..., channel, *spatial]
        The data tensor with leading batch dimension, the followed by channels
        and trailing spatial dimensions.

    signal_ndim : int, default=1
        The number of dimensions in each signal. Can only be 1, 2 or 3.

    complex : bool, default=True
        Whether to return real and imaginary parts of the complex Fourier
        coefficients or compute their absolute values (complex modulus).

    shift : bool, default=False
        Whether to shift the zero-frequency component to the center of the
        multidimensional spectrum.

    Returns
    -------
    spectrum : tensor
        The resulting Fourier tensor. If `complex` is True, then the channel
        dimension is doubled: `real` component of the Fourier coefficients of
        all channels are followed by the `imaginary` components. If `complex`
        is False, then the channel dimension is not doubled.

    Details
    -------
    Fourier transform is applied to each channel separately. Ensures that
    the result is c-contiguous.
    """

    # expects `input` of shape [*batch] x chan x [*signal_dim]
    assert input.dim() >= 1 + signal_ndim

    # ... x chan x [*signal_dim] -> (z) ... x chan x [*signal_dim] x 2
    z = torch_rfft(input, normalized=False, signal_ndim=signal_ndim)
    # fft of a real signal is conj-symmteric: z[j] = z[n-j].conj()

    # roll cplx dim (2) in front of channel and signal dim
    # z -> [*batch] x 2 x chan x [*signal_dim]
    n_last = z.dim() - 1
    zdim = n_last - (signal_ndim + 1)
    z = z.permute(*range(zdim), n_last, *range(zdim, n_last))

    if shift:
        # Shift the 0Hz component of [*signal_dim] to the center
        z = torch_fftshift(z, dims=tuple(range(-signal_ndim, 0)))

    if complex:
        data = z.contiguous()

    else:
        # the absolute value of a complex number
        data = torch.norm(z, p=2, dim=zdim, keepdim=True)

    # collapse zdim and chan (re-im concatenated, maintains contiguity)
    return data.flatten(zdim, zdim + 1)


class FeedFourierFeatures(BaseFeedWrapper):
    def __init__(self, feed, signal_ndim=1, shift=False, cplx=True):
        super().__init__(feed)
        self.signal_ndim, self.shift, self.cplx = signal_ndim, shift, cplx

    def __iter__(self):
        return self.iter_impl(self.feed, self.signal_ndim, self.cplx, self.shift)

    @classmethod
    def iter_impl(cls, feed, signal_ndim=1, cplx=True, shift=False):
        """Fourier-transform the first item in the batch tuple.

        Parameters
        ----------
        feed : iterable
            The data loader instance to draw batches from.

        signal_ndim : int, default=1
            The number of dimensions in each signal. Can only be 1, 2 or 3.

        complex : bool, default=True
            Whether to return real and imaginary parts of the complex Fourier
            coefficients or compute their absolute values (complex modulus).

        shift : bool, default=False
            Whether to shift the zero-frequency component to the center of the
            multidimensional spectrum.

        Yields
        ------
        batch : iterable
            An iterable that constitutes the batch.

        Details
        -------
        For details see `to_fourier`.
        """
        for signal, *rest in feed:
            if signal.dim() == 1 + signal_ndim:
                signal = signal.unsqueeze(-1 - signal_ndim)

            yield [to_fourier(signal, signal_ndim, cplx, shift), *rest]
