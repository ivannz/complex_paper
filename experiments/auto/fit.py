import tqdm
import torch
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from .delayed import DelayedKeyboardInterrupt
from .utils import state_dict_to_cpu, collate_history


def fit_one_epoch(model, objective, feed, optim, grad_clip=0., callback=None):
    """Run one SGD pass over the data feed for the objective and the model
    using the optimizer, and optionally clipping the gradients.

    Parameters
    ----------
    model : torch.nn.Module
        The updated instance of the model to fit.

    objective : BaseObjective
        The objective function to optimize, aka regularized loss.

    feed : batch iterator
        The batch generator to use for SGD in each epoch.

    optim : torch.optim.optimizer.Optimizer
        The optimizer to use during SGD loop. Supposed to be initialized with
        the provided `model`.

    grad_clip : float, default=0.
        The radius of the L2 ball to clip grads to. Disabled if 0.

    callback : callable, default=None
        Called after each SGD step, this function receives the loss components,
        including the norm of the gradient, if `grad_clip` is positive.

    Returns
    -------
    losses : list of float
        The list of the running loss values.

    Details
    -------
    Updates the internal states of the model and optimizer.
    """
    model.train()
    losses, grad_norm = [], np.nan
    for data, target in feed:
        with DelayedKeyboardInterrupt("delay"):  # fire after SGD step
            optim.zero_grad()  # clear anywhere before `.backward`

            # Compute the composite objective, creating the autograd graph
            loss = objective(model, data, target)

            loss.backward()
            if grad_clip > 0:
                grad_norm = clip_grad_norm_(model.parameters(), grad_clip)

            #  https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            optim.step()

        losses.append(float(loss))
        if callable(callback):
            # track zero-th parameter group's learning rate
            lrs = [group.get("lr", np.nan) for group in optim.param_groups]
            callback((*objective.component_values_, grad_norm, lrs[0]))

        # abort on nan -- no need to waste compute
        if np.isnan(losses[-1]):
            raise FloatingPointError

    return losses


def fit(model, objective, feed, optim, *, sched=None, early=None,
        n_epochs=100, grad_clip=0., verbose=True):
    """Fit a model to the objective on the data feed for specified number of
    epochs with optimizer, lr-schedule and gradient clipping.

    Parameters
    ----------
    model : torch.nn.Module
        The updated instance of the model to fit.

    objective : BaseObjective
        The objective function to optimize, aka regularized loss.

    feed : batch iterator
        The batch generator to use for SGD in each epoch.

    optim : torch.optim.optimizer.Optimizer
        The optimizer to use during SGD loop. Supposed to be initialized with
        the provided `model`.

    sched : torch.optim.ls_scheduler.*, default=None
        The learning rate schedule to use after each epoch. Expected to be
        already initialized to the provided `optim`.

    early : EarlyStopping callback, default=None
        An object implementing the early fit termination mechanics, based on
        the performance on a held out dataset.

    n_epoch : int, default=100
        The number of passes over the provided `feed`.

    grad_clip : float, default=0.
        The radius of the L2 ball to clip grads to. Disabled if 0.

    verbose : bool, default=True
        Whether to print a progress bar with current learning information.

    Returns
    -------
    model : torch.nn.Module
        The updated instance of the model.

    emergency : bool
        Boolean indicating if either a keyboard interrupt took place, a NAN
        was encountered during fit, or the loop was otherwise aborted.

    history : dict
        A dictionary to the tracked loss components, including the norm of the
        gradient, if `grad_clip` is positive.

    Details
    -------
    Forces the model in `train` mode before the nested SGD loop and forces it
    into `eval` mode afterwards.
    """
    model.train()
    history, model_backup = [], {}
    with tqdm.tqdm(range(n_epochs), disable=not verbose) as bar:
        def history_append(values):
            history.append(values)
            if verbose:
                # format the components of the loss objective
                *terms, grad_norm, lr = map("{:.2e}".format, values)
                status = repr(terms).replace("'", "")
                bar.set_postfix_str(f"{status} |g| {grad_norm} lr {lr}")

        # try-catch for graceful return
        try:
            for epoch in bar:
                # checkpointer and early stopper steps
                if early is not None:
                    early.step(epoch)

                model_backup = state_dict_to_cpu(model.state_dict())
                epoch_loss = fit_one_epoch(model, objective, feed, optim,
                                           grad_clip=grad_clip,
                                           callback=history_append)

                # scheduler step: the `.step` api isn't standardized
                if sched is not None:
                    if isinstance(sched, ReduceLROnPlateau):
                        sched.step(np.mean(epoch_loss))

                    else:
                        sched.step()

        except FloatingPointError as e:  # thrown by fit_one_epoch
            model.load_state_dict(model_backup)
            emergency = e

        except KeyboardInterrupt as e:
            emergency = e

        except StopIteration as e:  # thrown by early stopper
            emergency = None  # e  # Early Stopping is not an emergency

        else:  # no exception raised, no loop broken out of -- no emergency
            emergency = None

    # Collect histories of objective's components and the norm of the gradient
    history = collate_history(history, [*objective.terms, "|g|", "lr"])

    model.eval()
    return model, emergency, history
