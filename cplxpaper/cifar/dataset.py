from torchvision import datasets, transforms


class BaseCIFAR10(datasets.CIFAR10):
    """Base dataset class for CIFAR10.

    Details
    -------
    Taken from the tutorial on classification in pytorch.
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self, root, train=True):
        super().__init__(root, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]), train=train, download=True)


class CIFAR10Train(BaseCIFAR10):
    def __init__(self, root):
        super().__init__(root, train=True)


class CIFAR10Test(BaseCIFAR10):
    def __init__(self, root):
        super().__init__(root, train=False)
