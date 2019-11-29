#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tqdm
import h5py
import numpy as np

import torch
import torch.utils.data

from ncls import NCLS64
from resampy import resample
from bisect import bisect_right

from numpy.lib.stride_tricks import as_strided


def resample_h5(file_in, file_out, frame_rate_in, frame_rate_out, keys=None):
    # Authors: Dmitriy Serdyuk
    # (2019-11-24) changed to hdf5 supports
    print(f""".. resampling {file_in} ({frame_rate_in}Hz) """
          f"""into {file_out} ({frame_rate_out}Hz)""", flush=True)

    with h5py.File(file_in, "r") as h5_in, \
            h5py.File(file_out, "w") as h5_out:

        ratio = frame_rate_out / float(frame_rate_in)

        # pour from h5_in to h5_out
        items = h5_in.items()
        if keys is not None:
            items = [(k, h5_in[k]) for k in keys]

        for key, group in tqdm.tqdm(items):

            # resample the waveform
            data = resample(np.array(group["data"]),
                            frame_rate_in, frame_rate_out)
            h5_out.create_dataset(f"{key}/data", data=data)

            # change start-end times
            labels = np.array(group["labels"])
            labels["start_time"] = (labels["start_time"] * ratio).astype(int)
            labels["end_time"] = (labels["end_time"] * ratio).astype(int)
            h5_out.create_dataset(f"{key}/labels", data=labels)

    return file_out


class MusicNetHDF5(torch.utils.data.Dataset):
    """Dataset access to MusicNet sotred in HDF5 file.

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
        The data typee of the waveform windows and targets. Defautls ot float32
        for easier compatibility with torch.
    """

    def __init__(self, hdf5, window=4096, stride=512, at=None, dtype=np.float32):
        # ensure an open HDF5 handle
        assert isinstance(hdf5, h5py.File) and hdf5.id
        # assumes note_ids 21..105, i.e. 84 class labels

        # build object and label lookup
        indptr, references = [0], []
        for ix, (key, group) in enumerate(hdf5.items()):
            obj, label = group["data"], group["labels"]

            # construct a fast lookup of music notes: a Nested
            #  Containment List is much faster than Interval Tree.
            tree = NCLS64(np.int64(label["start_time"]),    # start
                          np.int64(label["end_time"]),      # end
                          np.int64(label["note_id"] - 21))  # note - 21 (base)

            # cache hdf5 objects (stores references to hdf5 objects!)
            references.append((obj, tree))

            # the number of full valid windows fitting into the signal
            # (TODO) initial offset and padding
            strided_size = ((len(obj) - window + 1) + stride - 1) // stride
            indptr.append(indptr[-1] + strided_size)

        self.n_samples, self.n_outputs = indptr[-1], 84
        self.objects, self.labels = zip(*references)
        self.indptr = tuple(indptr)

        self.keys = dict(zip(hdf5.keys(), range(len(self.objects))))
        self.limits = dict(zip(hdf5.keys(), zip(indptr, indptr[1:])))

        # midpoint by default
        at = (window // 2) if at is None else at
        self.at = (window + at) if at < 0 else at

        self.hdf5, self.window, self.stride = hdf5, window, stride
        self.dtype = dtype

    def __getitem__(self, index):
        index = self.n_samples + index if index < 0 else index
        key = bisect_right(self.indptr, index) - 1  # a[:k] <= v < a[k:]
        if not (0 <= key < len(self.objects)):
            raise KeyError

        ix = (index - self.indptr[key]) * self.stride

        # fetch data and construct labels: random access is slow
        obj, lab = self.objects[key], self.labels[key]
        data = obj[ix:ix + self.window].astype(self.dtype)

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
        if (stride > 0 and i0 >= i1) or (stride < 0 and i0 <= i1):
            return np.empty((0, self.window), self.dtype)

        # read a contiguous slice from the dataset (waveform)
        chunks = []
        base, k0 = -1, bisect_right(self.indptr, i0) - 1
        for ii in range(i0, i1, stride):
            if stride > 0 and self.indptr[k0 + 1] <= ii:
                base, k0 = -1, k0 + 1

            elif stride < 0 and ii < self.indptr[k0]:
                base, k0 = -1, k0 - 1

            if base < 0:
                # h5py caches chunks, so adjacent reads are not reloaded.
                obj, base = self.objects[k0], self.indptr[k0]

            ix = (ii - base) * self.stride
            chunks.append(obj[ix:ix + self.window].astype(self.dtype))

        # Making a strided view us pointless since slices
        #  across boundaries would have to be collated anyway.
        return np.stack(chunks, axis=0)

    def fetch_data_key(self, key):
        if key not in self.keys:
            raise KeyError(f"`{key}` not found")

        key = self.keys[key]
        data = self.objects[key][:].astype(self.dtype)

        *head, stride = data.strides
        stride = *head, self.stride * stride, stride

        beg, end = self.indptr[key], self.indptr[key+1]
        shape = *data.shape[:-1], end - beg, self.window

        return as_strided(data, shape, stride, writeable=False)
