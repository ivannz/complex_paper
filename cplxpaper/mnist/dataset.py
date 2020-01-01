"""MNIST-like datasets used in experiments.

Important
---------
The roles of train and test are intentionally swapped! The reason being to
be able to test the if the subnetwork, found by variational dropout and/or
automatic relevance determination methods (both real and complex valued) has
any generalization ability.

PS
--
Maybe it was better to use a small part of the official TRAIN samples?
Without this hassle and role-switching?
"""

from torchvision import datasets, transforms


class EMNIST(datasets.EMNIST):
    """EMNIST dataset class with updated URL (2020-01-01).

    Intentionally uses the same class/type name as `datasets.EMNIST`.
    """
    url = """https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download"""


class EMNIST_Letters(object):
    """the `Letters` split of the EMNIST Dataset.

    See https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Details
    -------
    I use __new__ below to prevent downloading the same dataset multiple
    times due to differences in derived class names (the usual inheritance).

    Another reason for __new__ is for simpler string-based importing.
    """

    def __new__(cls, root, train=True, transform=None,
                target_transform=None, download=False):

        return EMNIST(root, split="letters", transform=transform, train=train,
                      target_transform=target_transform, download=download)


def get_dataset(cls, root, train=True):
    """Obtain an object representing the specified MNIST-like dataset."""
    assert cls in (datasets.MNIST, datasets.KMNIST,
                   EMNIST_Letters, datasets.FashionMNIST)

    # MNIST-like are multiclass datasets with grayscale 28x28 images:
    #  x is float-tensor with shape [..., 1, 28, 28], y is long-tensor labels
    return cls(root, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]), train=train, download=True)


class EMNIST_Letters_Train(object):
    """Train sample from the EMNIST letters dataset used in these experiments.

    Details
    -------
    This is actually the official `TEST` part of `EMNIST letters` dataset.
    It is used for TRAINING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(EMNIST_Letters, root, train=False)


class EMNIST_Letters_Test(object):
    """Test sample from the EMNIST letters dataset used in these experiments.

    Details
    -------
    This is actually the official `TRAIN` part of `EMNIST letters` dataset.
    It is used for TESTING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(EMNIST_Letters, root, train=True)


class MNIST_Train(object):
    """Train sample from the MNIST dataset used in these experiments.

    Details
    -------
    This is actually the official `TEST` part of `MNIST` dataset.
    It is used for TRAINING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(datasets.MNIST, root, train=False)


class MNIST_Test(object):
    """Test sample from the MNIST dataset used in these experiments.

    Details
    -------
    This is actually the official `TRAIN` part of `MNIST` dataset.
    It is used for TESTING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(datasets.MNIST, root, train=True)


class KMNIST_Train(object):
    """Train sample from the Kuzushiji-MNIST dataset used in these experiments.
    https://github.com/rois-codh/kmnist

    Details
    -------
    This is actually the official `TEST` part of `KMNIST` dataset.
    It is used for TRAINING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(datasets.KMNIST, root, train=False)


class KMNIST_Test(object):
    """Test sample from the Kuzushiji-MNIST dataset used in these experiments.

    Details
    -------
    This is actually the official `TRAIN` part of `KMNIST` dataset.
    It is used for TESTING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(datasets.KMNIST, root, train=True)


class FashionMNIST_Train(object):
    """Train sample from the Fashion-MNIST dataset used in these experiments.

    Details
    -------
    This is actually the official `TEST` part of `Fashion-MNIST` dataset.
    It is used for TRAINING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(datasets.FashionMNIST, root, train=False)


class FashionMNIST_Test(object):
    """Test sample from the Fashion-MNIST dataset used in these experiments.

    Details
    -------
    This is actually the official `TRAIN` part of `Fashion-MNIST` dataset.
    It is used for TESTING intentionally.
    """

    def __new__(cls, root):
        return get_dataset(datasets.FashionMNIST, root, train=True)
