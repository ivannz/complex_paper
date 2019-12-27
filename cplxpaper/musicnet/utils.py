#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tqdm
import h5py
import numpy as np

from resampy import resample


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
