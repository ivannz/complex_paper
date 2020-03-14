from torch.optim.lr_scheduler import LambdaLR


class Trabelsi2017LRSchedule(LambdaLR):
    def __init__(self, optimizer):
        super().__init__(optimizer, self.lr_lambda)

    @classmethod
    def lr_lambda(cls, epoch):
        """Multiplicative LR schedule (for base lr 1e-3).
        
        Details
        -------
        Taken verbatim from https://github.com/ChihebTrabelsi/deep_complex_networks/blob/f7065cea654c8b2a049c40fcc40fd76817c15096/musicnet/scripts/train.py#L32-L53
        """
        if epoch < 10:
            return 1e-0  # (1e-3)

        elif epoch < 100:
            return 1e-1  # (1e-4)

        elif epoch < 120:
            return 5e-2  # (5e-5)

        elif epoch < 150:
            return 1e-2  # (1e-5)

        else:
            return 1e-3  # (1e-6)


class FastStepScheduler(LambdaLR):
    def __init__(self, optimizer):
        super().__init__(optimizer, self.lr_lambda)

    @classmethod
    def lr_lambda(cls, epoch):
        """Faster lr annealing for base lr 1e-3."""
        if epoch < 5:
            return 1e-0  # (1e-3)

        elif epoch < 10:
            return 1e-1  # (1e-4)

        elif epoch < 20:
            return 5e-2  # (5e-5)

        elif epoch < 50:
            return 1e-2  # (1e-5)

        else:
            return 1e-3  # (1e-6)
