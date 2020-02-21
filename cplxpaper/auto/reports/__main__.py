import os
import re
import json
import tqdm
import time
import pickle
import argparse

from multiprocessing import Process, Array, Semaphore
from multiprocessing import JoinableQueue, Queue
from threading import Thread

from itertools import chain


def analyze_experiment(kind, experiment, *, device):
    # use process-local imports to make sure randomness and contexts are
    #  initialized afresh

    # from cplxpaper.auto.parameter_grid import reconstruct_grid
    if kind == "trade-off":
        from cplxpaper.auto.reports.tradeoff import evaluate_experiment
        result = evaluate_experiment(experiment, device=device)

    elif kind == "debug":
        # process-local import
        # import numpy
        import torch

        # delay = float(numpy.random.rand(1))
        delay = float(torch.rand(1, device=torch.device(device)).cpu())

        # time.sleep(max(1 - experiment / 10000, 0))
        time.sleep(delay)
        result = kind, experiment, device, delay

    else:
        raise ValueError(f"Unrecognized report `{kind}`.")

    return result


class Budget(object):
    """Device budget (for lack of a better term)

    Details
    -------
    Keeps a semaphore for the total number of resources available,
    and a device-specific atomic counter of available slots.
    """
    def __init__(self, devices, n_per_device):
        self.devices, self.n_per_device = devices, n_per_device
        self.total = Semaphore(len(devices) * n_per_device)
        self.alloc = Array("i", len(devices) * [n_per_device])

    def acquire(self):
        self.total.acquire()
        with self.alloc.get_lock():
            # get the largest counter
            index = max(range(len(self.alloc)), key=self.alloc.__getitem__)
            # assert self.alloc[index] > 0

            # acquire the index and decrease the counter
            self.alloc[index] -= 1
        return index

    def release(self, index):
        with self.alloc.get_lock():
            self.alloc[index] += 1
        self.total.release()

    def __getitem__(self, index):
        return self.devices[index]


def store(queue, report, append):
    """Store objects streamed through a queue in a binary file.

    Details
    -------
    Using a `terminated` flag is problematic, since the queue blocks
    on `.get()` if empty, but it is entirely possible for the jobs to
    be so heavy and log, that the queue starves.

    We could use wait for a `.get()` on a closed queue, because semantically
    `.close()` also indicates that the queue no longer accepts new results
    (see the docs for `multiprocessing`). However this is not so, and a closed
    queue is closed entirely and cannot be read from.

    Using a special sentinel value not elegant, but is the most viable option.
    """
    with open(report, "ab" if append else "wb") as storage:
        result = queue.get()
        while result is not None:
            storage.write(pickle.dumps(result))

            # `.get()` on a closed queue raises `EOFError`, otherwise
            # blocks if empty
            result = queue.get()


def worker(wid, kind, index, jobs, budget, output):
    """Wait until a job in the queue appears and perform it."""
    job = jobs.get()
    while job is not None:
        try:
            output.put_nowait((
                wid, analyze_experiment(kind, job, device=budget[index])
            ))

        finally:
            jobs.task_done()
            budget.release(index)

        job = jobs.get()
    jobs.task_done()


def verify_experiment(folder):
    """Check if the experiment at the specified folder has been completed."""
    if not os.path.isdir(folder):
        return False

    filename = os.path.join(folder, "config.json")
    if not os.path.isfile(filename):
        return False

    with open(filename, "r") as fin:
        manifest = json.load(fin)

    # stage map to check if any one is missing
    folder, _, filenames = next(os.walk(folder))
    stages = dict.fromkeys(manifest["stage-order"], False)
    for j, stage in enumerate(stages.keys()):
        # synchronized with naming format at ../auto.py#L393
        pat = re.compile(f"^{j}-{stage}\\s+.*\\.gz$")
        match = next(filter(None, map(pat.match, filenames)), None)
        stages[stage] = match is not None

    return all(stages.values())


def enumerate_experiments(grid):
    """Iterate over all complete experiments in a grid."""
    grid = os.path.abspath(os.path.normpath(grid))
    if not os.path.isdir(grid):
        return

    grid, _, filenames = next(os.walk(grid))
    for name, ext in map(os.path.splitext, filenames):
        if ext != ".json" or name.startswith("."):
            continue

        experiment = os.path.join(grid, name)
        if not verify_experiment(experiment):
            continue

        yield experiment


def main(*, paths, kind, report, append=False,
         devices=("cuda:1", "cuda:3"), per_device=1):
    if os.path.isfile(report) and not append:
        raise RuntimeError(f"Refusing to overwrite existing `{report}`")

    # create a thread, to collect and commit results into storage
    output = Queue()
    collector = Thread(target=store, args=(output, report, append))

    # each device is associated with its own process pool
    queues, pool, budget = [], [], Budget(devices, per_device)
    for index, device in enumerate(devices):
        queues.append(JoinableQueue(per_device))
        for _ in range(per_device):
            pool.append(Process(target=worker, args=(
                len(pool), kind, index,
                queues[-1], budget, output
            )))

    # start everyone
    collector.start()
    for p in pool:
        p.start()

    # main job queue
    # gather and enqueue all experiments from the folder
    experiments = chain(*map(enumerate_experiments, paths))
    for experiment in tqdm.tqdm(experiments, disable=False):
        queues[budget.acquire()].put_nowait(experiment)

    # terminated subprocesses by signaling empty job
    for q in queues:
        for _ in range(per_device):
            q.put(None)
        q.join()

    for p in pool:
        p.join()

    # all jobs are done once we are here, so join the collector thread
    output.put(None)
    collector.join()

    print(f">>> complete `{report}`")
    return report


parser = argparse.ArgumentParser(description='Auto report generation',
                                 add_help=True)

parser.add_argument(
    '--devices', type=str, nargs='+', required=False,
    default=("cuda:1", "cuda:3"), help='allowed devices')
parser.add_argument(
    '--per-device', type=int, required=False, default=1,
    help='the maximum number of jobs per device')

parser.add_argument(
    '--append', action='store_true', required=False,
    default=False, help='Append to report, instead of creating new one')

parser.add_argument(
    'kind', type=str, help='the type of report to preprocess results for')
parser.add_argument(
    'report', type=str, help='the name of the report pickle')
parser.add_argument(
    'paths', type=str, nargs='+',
    help='path to grids or particular experiment results')

main(**vars(parser.parse_args()))
