#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tqdm
import h5py
import numpy as np

import torch
import torch.utils.data

# from functools import lru_cache

from ncls import NCLS64
from intervaltree import IntervalTree
from resampy import resample


def resample_h5(file_in, file_out, frame_rate_in, frame_rate_out):
    # Authors: Dmitriy Serdyuk
    # (2019-11-24) changed to hdf5 supports
    print(f""".. resampling {file_in} ({frame_rate_in}Hz) """
          f"""into {file_out} ({frame_rate_out}Hz)""", flush=True)

    with h5py.File(file_in, "r") as h5_in, \
            h5py.File(file_out, "w") as h5_out:

        ratio = frame_rate_out / float(frame_rate_in)

        # pour from h5_in to h5_out
        for key, group in tqdm.tqdm(h5_in.items()):

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


class MuscNetHDF5(torch.utils.data.Dataset):
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
    """
    def __init__(self, hdf5, window=4096, stride=512, verbose=False):
        # ensure an open HDF5 handle
        assert isinstance(hdf5, h5py.File) and hdf5.id

        # has to get note_ids from somewhere
        note_ids = set()  # OVERRIDE!

        # build data lookup and label tree (O(n log n))
        self.intervals, self.labels, self.total = IntervalTree(), {}, 0
        for key, group in tqdm.tqdm(hdf5.items(), disable=not verbose,
                                    desc="building lookup"):
            obj, lab = group["data"], group["labels"]

            # cache hdf5 group objects (stores references to hdf5 objects!)
            strided_size = ((len(obj) - window + 1) + stride - 1) // stride
            self.intervals[self.total:self.total + strided_size] = key, obj
            self.total += strided_size

            # collect notes
            note_ids.update(lab["note_id"])

            # construct a fast lookup of music notes: a Nested Containment List
            #  is much faster than Interval Tree.
            self.labels[key] = NCLS64(
                np.array(lab["start_time"], dtype=np.int64),  # start
                np.array(lab["end_time"], dtype=np.int64),    # end
                np.array(lab["note_id"], dtype=np.int64))     # payload id

        # preallocate `note_id` one-hots
        self.onehots, self.n_classes = {}, max(note_ids) - min(note_ids) + 1
        eye = np.eye(self.n_classes, dtype=np.float32)
        for j, note in enumerate(sorted(note_ids)):
            self.onehots[note] = eye[j]

        self.hdf5, self.window, self.stride = hdf5, window, stride

    def __getitem__(self, index):
        index = self.total + index if index < 0 else index

        # raises KeyError if the query is empty
        interval = self.intervals.at(index).pop()
        ix = (index - interval.begin) * self.stride

        # fetch data and construct labels: random access is slow
        key, obj = interval.data
        data = obj[ix:ix + self.window]

        midp = ix + self.window // 2
        labels = self.labels[key].find_overlap(midp, midp + 1)
        onehot = sum((self.onehots[note_id] for a, b, note_id in labels),
                     np.zeros(self.n_classes, dtype=np.float32))

        return data, onehot

    def __len__(self):
        return self.total
