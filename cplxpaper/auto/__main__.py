import os
import tqdm
import json
import time
import argparse

from multiprocessing import Process, JoinableQueue, Array, Semaphore


def one_experiment(wid, device, manifest):
    # Import torch inside this function, because it is called in a subprocess.
    # This makes sure that torch's RNG context is initialized afresh.
    from cplxpaper.auto import run
    # from cplxpaper.auto.auto import debug as run

    # open manifest
    options = json.load(open(manifest, "r"))
    options["device"] = device

    # create the target folder
    target, ext = os.path.splitext(manifest)
    # name = os.path.basename(target)

    # run the experiment
    flag = os.path.join(target, "INCOMPLETE")
    if not os.path.isdir(target) or os.path.isfile(flag):
        os.makedirs(target, exist_ok=True)  # once created it is kept

        open(flag, "wb").close()  # set busy flag
        # print(f">>> {wid:03d}-{device} {name}")
        run(options, target, time.strftime("%Y%m%d-%H%M%S"), verbose=False)
        os.remove(flag)  # if run raises the flag is not reset


class Budget(object):
    """Device budget (for lack of a better term)

    Details
    -------
    Keeps a semaphore for the total number of resources available, and
    a device-specific atmoic counter of available slots.
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


def worker(wid, index, jobs, budget):
    """Wait until a job in the queue appears and perform it."""
    job, n_retries = jobs.get()
    while job is not None:
        try:
            rescheduled = False
            one_experiment(wid, budget[index], job)

        except KeyboardInterrupt:
            break  # abort, and exit this thread

        except Exception as e:
            # gobble up any exception, and reschedule if necessary
            if n_retries > 0:
                # print(f"reschedule due to {type(e).__name__}({str(e)})")
                jobs.put_nowait((job, n_retries - 1))
                rescheduled = True

        finally:
            jobs.task_done()
            if not rescheduled:
                budget.release(index)

        job, n_retries = jobs.get()
    jobs.task_done()


def main(path, devices=("cuda:1", "cuda:3"), n_per_device=1, n_retries=1):
    # each device is associated with its own process pool
    queues, pool, budget = [], [], Budget(devices, n_per_device)
    for index, device in enumerate(devices):
        queues.append(JoinableQueue(n_per_device))
        for _ in range(n_per_device):
            pool.append(Process(target=worker, args=(
                len(pool), index, queues[-1], budget
            )))

    # start everone
    for p in pool:
        p.start()

    # gather and enqueue all manifests from the folder
    for name, ext in map(os.path.splitext, tqdm.tqdm(os.listdir(path))):
        if ext != ".json" or name.startswith("."):
            continue

        experiment = os.path.join(path, name)
        flag = os.path.join(experiment, "INCOMPLETE")
        if not os.path.isdir(experiment) or os.path.isfile(flag):
            # a job is a manifset and retry counter (incl. zero-th retry)
            index = budget.acquire()
            queues[index].put_nowait((experiment + ".json", n_retries))

    # terminated subprocesses by signalling empty job
    for q in queues:
        for _ in range(n_per_device):
            q.put((None, 0))
        q.join()

    for p in pool:
        p.join()

    print(">>> complete")


parser = argparse.ArgumentParser(description='Auto experiment runner',
                                 add_help=True)

parser.add_argument(
    '--devices', type=str, nargs='+', required=False,
    default=("cuda:1", "cuda:3"), help='allowed devices')
parser.add_argument(
    '--per-device', type=int, required=False, default=1,
    help='the maximum number of jobs per device')

parser.add_argument(
    'path', type=str, help='the folder with experiment manifests')

# parser.add_argument('--save-optim', dest='save_optim', action='store_true')
# parser.add_argument('--no-save-optim', dest='save_optim', action='store_false')
# parser.set_defaults(save_optim=True)

args = parser.parse_args()

path = args.path
if os.path.exists(path) and os.path.isdir(path):
    main(path, args.devices, args.per_device)
