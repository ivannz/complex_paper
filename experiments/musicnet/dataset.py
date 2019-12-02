import h5py
import numpy as np

import torch
import torch.utils.data

from ncls import NCLS64
from bisect import bisect_right

from numpy.lib.stride_tricks import as_strided


class MusicNetHDF5(torch.utils.data.Dataset):
    """Dataset access to MusicNet stored in HDF5 file.

    Parameters
    ----------
    hdf5 : h5py.File
        Open HDF5 file handle.

    window : int, or tuple
        The size of the window. If tuple, then specifies the number of samples
        before `t` and after.

    stride : int, default=1
        The stride of the sliding window.

    at : int, or None, default=None
        The index within the `window` at which the targets are collected.
        Should be thought of as the offset to time `t` in the window. Midpoint
        of the window by default (None).

    dtype : np.dtype, default=np.float32
        The data type of the waveform windows and targets. Defaults to float32
        for easier compatibility with torch.

    resident : bool, default=False
        Wheter to cache the raw waveform data into ram on init.
    """

    def __init__(self, hdf5, window=4096, stride=512, at=None,
                 dtype=np.float32, resident=False):
        # ensure an open HDF5 handle
        assert isinstance(hdf5, h5py.File) and hdf5.id
        # assumes note_ids 21..105, i.e. 84 class labels
        self.n_outputs, base_note_id = 84, 21
        self.probability = np.zeros((len(hdf5), self.n_outputs))

        # build object and label lookup
        indptr, references = [0], []
        for ix, (key, group) in enumerate(hdf5.items()):
            obj, label = group["data"], group["labels"]

            # construct a fast lookup of music notes: a Nested
            #  Containment List is much faster than Interval Tree.
            note_id = label["note_id"] - base_note_id
            start_time, end_time = label["start_time"], label["end_time"]
            tree = NCLS64(np.int64(start_time), np.int64(end_time),
                          np.int64(note_id))

            # Estimate the probability of a musical note playing
            #  at a randomly picked sample within the composition
            counts = np.bincount(note_id, minlength=self.n_outputs,
                                 weights=end_time - start_time)
            self.probability[ix] = counts / float(len(obj))

            # cache hdf5 objects (stores references to hdf5 objects!)
            if resident:
                obj = obj[:].astype(dtype, copy=True)  # read the data
            references.append((obj, tree))

            # the number of full valid windows fitting into the signal
            # (TODO) initial offset and padding
            strided_size = ((len(obj) - window + 1) + stride - 1) // stride
            indptr.append(indptr[-1] + strided_size)

        self.n_samples, self.indptr = indptr[-1], tuple(indptr)
        self.objects, self.labels = zip(*references)

        self.keys = dict(zip(hdf5.keys(), range(len(hdf5))))
        self.limits = dict(zip(hdf5.keys(), zip(indptr, indptr[1:])))

        # midpoint by default
        at = (window // 2) if at is None else at
        self.at = (window + at) if at < 0 else at

        self.hdf5, self.window, self.stride = hdf5, window, stride
        self.dtype = dtype

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.fetch_data_slice(index)

        elif not isinstance(index, int):
            return self.fetch_data_key(index)

        index = self.n_samples + index if index < 0 else index
        key = bisect_right(self.indptr, index) - 1  # a[:k] <= v < a[k:]
        if not (0 <= key < len(self.objects)):
            raise KeyError

        ix = (index - self.indptr[key]) * self.stride

        # fetch data and construct labels: random access is slow
        obj, lab = self.objects[key], self.labels[key]
        data = obj[ix:ix + self.window].astype(self.dtype, copy=False)

        notes = np.zeros(self.n_outputs, self.dtype)
        for b, e, j in lab.find_overlap(ix + self.at, ix + self.at + 1):
            notes[j] = 1

        return data, notes

    def __len__(self):
        return self.n_samples

    def __dir__(self):
        return list(self.keys)

    def fetch_data_slice(self, index):
        assert isinstance(index, slice)

        i0, i1, stride = index.indices(self.n_samples)
        # Such n, that i0 .. i0 + n s < i1 <= i0 + (n+1) s when s > 0, or
        #  i0 .. i0 + n s > i1 >= i0 + (n+1) s if s < 0.
        size = (i1 - i0 + stride + (+1 if stride < 0 else -1)) // stride
        if size <= 0:
            return np.empty((0, self.window), self.dtype)

        # Making a strided view us pointless, since slices across boundaries
        #  would have to be collated anyway
        chunks = np.empty((size, self.window), self.dtype)

        # read a contiguous slice from the dataset (waveform)
        base, k0 = -1, bisect_right(self.indptr, i0) - 1
        for j, ii in enumerate(range(i0, i1, stride)):
            if stride > 0 and self.indptr[k0 + 1] <= ii:
                base = -1
                while self.indptr[k0 + 1] <= ii:
                    k0 += 1

            elif stride < 0 and ii < self.indptr[k0]:
                base = -1
                while ii < self.indptr[k0]:
                    k0 -= 1

            if base < 0:
                # h5py caches chunks, so adjacent reads are not reloaded.
                obj, base = self.objects[k0], self.indptr[k0]

            ix = (ii - base) * self.stride
            chunks[j] = obj[ix:ix + self.window]

        return chunks

    def fetch_data_key(self, key):
        if key not in self.keys:
            raise KeyError(f"`{key}` not found")

        key = self.keys[key]
        data = self.objects[key][:].astype(self.dtype, copy=False)

        *head, stride = data.strides
        stride = *head, self.stride * stride, stride

        beg, end = self.indptr[key], self.indptr[key+1]
        shape = *data.shape[:-1], end - beg, self.window

        return as_strided(data, shape, stride, writeable=False)

    def __repr__(self):
        return repr(self.hdf5)
