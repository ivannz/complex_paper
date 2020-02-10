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
    target, ext = os.path.splitext(manifest)
    name = os.path.basename(target)

    # run the experiment
    flag = os.path.join(target, "INCOMPLETE")
    if not os.path.isdir(target) or os.path.isfile(flag):
        os.makedirs(target, exist_ok=True)  # once created it is kept

        open(flag, "wb").close()  # set busy flag
        print(f">>> {wid:03d}-{device} {name}")
        run(options, target, time.strftime("%Y%m%d-%H%M%S"), verbose=False)
        os.remove(flag)  # if run raises the flag is not reset


def worker(wid, jobs, devarray, devices):
    """Poll the job queue and dispatch to a device."""
    job, n_retries = jobs.get()
    while job is not None:
        # Acquire the next free device for this job and make sure the
        #  resources are released.
        devid = acquire(devarray)
        try:
            one_experiment(wid, devices[devid], job)

        except Exception as e:
            # gobble up any exception, and reschedule if necessary
            if n_retries > 0:
                print(f"reschedule due to {type(e).__name__}({str(e)})")
                jobs.put_nowait((job, n_retries - 1))

        finally:
            release(devarray, devid)
            jobs.task_done()

        job, n_retries = jobs.get()

    jobs.task_done()


def main(path, devices=("cuda:1", "cuda:3"), n_per_device=1, n_retries=1):
    # create a pool of workers and a device availability array
    workers, manifests = [], JoinableQueue()
    devarray = Array("i", len(devices) * [n_per_device])
    for device in n_per_device * devices:
        p = Process(target=worker, args=(
            len(workers), manifests, devarray, devices
        ))
        p.start()
        workers.append(p)

    # gather and enqueue all manifests from the folder
    for name, ext in map(os.path.splitext, os.listdir(path)):
        if ext != ".json" or name.startswith("."):
            continue

        experiment = os.path.join(path, name)
        flag = os.path.join(experiment, "INCOMPLETE")
        if not os.path.isdir(experiment) or os.path.isfile(flag):
            # a job is a manifset and retry counter (incl. zero-th retry)
            manifests.put_nowait((experiment + ".json", n_retries))

    # enqueue `stop` signals as well
    for w in workers:
        # each worker immediately stops querying for a next job
        manifests.put((None, 0))

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

# parser.add_argument('--save-optim', dest='save_optim', action='store_true')
# parser.add_argument('--no-save-optim', dest='save_optim', action='store_false')
# parser.set_defaults(save_optim=True)

args = parser.parse_args()

path = args.path
if os.path.exists(path) and os.path.isdir(path):
    main(path, args.devices, args.per_device)
