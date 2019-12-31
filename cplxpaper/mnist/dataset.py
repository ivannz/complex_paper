from torchvision import datasets, transforms


class BaseMNIST(datasets.MNIST):
    """Base class for MNIST dataset."""

    def __init__(self, root, train=True):
        super().__init__(root, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]), train=train, download=True)


class MNISTTrain(BaseMNIST):
    def __init__(self, root):
        super().__init__(root, train=True)


class MNISTTest(BaseMNIST):
    def __init__(self, root):
        super().__init__(root, train=False)
