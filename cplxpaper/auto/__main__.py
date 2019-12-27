import os
import time
import json

import argparse

from multiprocessing import Process, JoinableQueue, Array


def acquire(array):
    """Acquire resources from a budget."""
    while True:
        with array.get_lock():
            # get the largest counter
            ix = max(range(len(array)), key=array.__getitem__)
            if array[ix] > 0:
                # acquire the index and decrease the counter
                array[ix] -= 1
                break

        # Busy wait on a next nonzero counter. This works fine if jobs are
        #  few and long. But there should be a better way with `asyncio`!
        pass
    return ix


def release(array, ix):
    """Release resources to the budget."""
    with array.get_lock():
        array[ix] += 1


def one_experiment(wid, device, manifest):
    # Import torch inside this function, because it is called in a subprocess.
    # This makes sure that torch's RNG context is initialized afresh.
    import torch
    from cplxpaper.auto import run

    # open manifest
    options = json.load(open(manifest, "r"))
    options["device"] = device

    # create the target folder
    path = os.path.dirname(manifest)
    name = os.path.basename(manifest)

    name, ext = os.path.splitext(name)
    target = os.path.join(path, name)

    # run the experiment
    if not os.path.exists(target):
        print(f">>> {wid:03d}-{device} {name}")
        run(options, target, time.strftime("%Y%m%d-%H%M%S"), verbose=False)


def worker(wid, jobs, devarray, devices):
    """Poll the job queue and dispatch to a device."""
    job = jobs.get()
    while job is not None:
        # Acquire the next free device for this job and make sure the
        #  resources are released.
        devid = acquire(devarray)
        try:
            one_experiment(wid, devices[devid], job)

        finally:
            release(devarray, devid)
            jobs.task_done()

        job = jobs.get()

    jobs.task_done()


def main(path, devices=("cuda:1", "cuda:3"), n_per_device=1):
    # create a pool of workers and a device availability array
    workers, manifests = [], JoinableQueue()
    devarray = Array("i", len(devices) * [n_per_device])
    for device in n_per_device * devices:
        p = Process(target=worker, args=(
            len(workers), manifests, devarray, devices
        ))
        p.start()
        workers.append(p)

    # gather and equeue all manifests from the folder
    for fn in os.listdir(path):
        if not fn.endswith(".json"):
            continue

        if fn.startswith("."):
            print(f"Not an experiment {fn}...", flush=True)
            continue

        manifests.put_nowait(os.path.join(path, fn))

    # enqueue `stop` signals as well
    for w in workers:
        # each worker immediately stops querying for a next job
        manifests.put(None)

    # wait for completion and stop workers
    manifests.join()
    for w in workers:
        w.join()

    print(">>> complete")


parser = argparse.ArgumentParser(description='Auto experiment runner',
                                 add_help=True)
parser.add_argument('path', type=str, help='the path to all manifests')
parser.add_argument('--devices', type=str, nargs='+',
                    default=("cuda:1", "cuda:3"), required=False,
                    help='allowed devices')
parser.add_argument('--per-device', type=int, default=1, required=False)

args = parser.parse_args()

path = args.path
if os.path.exists(path) and os.path.isdir(path):
    main(path, args.devices, args.per_device)
