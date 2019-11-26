#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tqdm
import h5py
import numpy as np

import torch
import torch.utils.data

from ncls import NCLS64
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
    """
    def __init__(self, hdf5, window=4096, stride=512):
        # ensure an open HDF5 handle
        assert isinstance(hdf5, h5py.File) and hdf5.id
        # assumes note_ids 21..105, i.e. 84 class labels

        # build object and label lookup
        intervals, n_samples = [], 0
        for ix, (key, group) in enumerate(hdf5.items()):
            obj, label = group["data"], group["labels"]

            # construct a fast lookup of music notes: a Nested
            #  Containment List is much faster than Interval Tree.
            tree = NCLS64(np.int64(label["start_time"]),    # start
                          np.int64(label["end_time"]),      # end
                          np.int64(label["note_id"] - 21))  # note - 21 (base)

            # cache hdf5 objects (stores references to hdf5 objects!)
            strided_size = ((len(obj) - window + 1) + stride - 1) // stride
            intervals.append((n_samples, n_samples + strided_size, obj, tree))

            n_samples += strided_size

        # build the object lookup
        beg, end, self.objects, self.labels = zip(*intervals)
        self.intervals = NCLS64(np.int64(beg), np.int64(end),
                                np.int64(np.r_[:len(self.objects)]))
        self.n_samples = n_samples

        self.hdf5, self.window, self.stride = hdf5, window, stride

    def __getitem__(self, index):
        index = self.n_samples + index if index < 0 else index
        interval = next(self.intervals.find_overlap(index, index + 1), None)
        if interval is None:
            raise KeyError

        beg, _, key = interval
        ix = (index - beg) * self.stride

        # fetch data and construct labels: random access is slow
        obj, lab = self.objects[key], self.labels[key]
        data = obj[ix:ix + self.window].astype(np.float32)

        midp = ix + self.window // 2
        notes = np.zeros(84, np.float32)  # n_classes assumption
        for b, e, j in lab.find_overlap(midp, midp + 1):
            notes[j] = 1

        return data, notes

    def __len__(self):
        return self.n_samples
