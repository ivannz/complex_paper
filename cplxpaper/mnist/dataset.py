"""MNIST-like datasets used in experiments.

Important
---------
The test splits of the original datasets are returned as is to make the results
in this paper comparable with performance reported elsewhere. However, the
train split is made intentionally smaller in order to be able to test the if
the sparse subnetwork, found by variational dropout and/or automatic relevance
determination methods (both real and complex valued) keeps it generalization
ability if at all.
"""
import torch

from torchvision import datasets, transforms
from torch.utils.data import Subset

from sklearn.model_selection import train_test_split


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


def stratified_split(dataset, train_size=None, test_size=None, random_state=None):
    """Random stratified train/test split."""
    targets = dataset.targets.cpu().numpy()
    ix_all = torch.arange(len(targets)).numpy()

    # use stratified split to get the required number of samples
    ix_train, ix_test = train_test_split(
        ix_all, stratify=targets, train_size=train_size, test_size=test_size,
        shuffle=True, random_state=random_state)

    return Subset(dataset, ix_train), Subset(dataset, ix_test)


def get_train_dataset(cls, root, train_size=None, random_state=None):
    """A stratified random subset of the `train` split of the original dataset.
    """
    dataset = get_dataset(cls, root, train=True)
    if train_size is None:
        return dataset

    train, valid = stratified_split(dataset, train_size=train_size,
                                    random_state=random_state)
    return train


class EMNIST_Letters_Train(object):
    """Train sample from the EMNIST letters dataset."""

    def __new__(cls, root, train_size=None, random_state=None):
        return get_train_dataset(EMNIST_Letters, root, train_size=train_size,
                                 random_state=random_state)


class EMNIST_Letters_Test(object):
    """Test sample from the EMNIST letters dataset."""

    def __new__(cls, root):
        return get_dataset(EMNIST_Letters, root, train=False)


class MNIST_Train(object):
    """Train sample from the MNIST dataset."""

    def __new__(cls, root, train_size=None, random_state=None):
        return get_train_dataset(datasets.MNIST, root, train_size=train_size,
                                 random_state=random_state)


class MNIST_Test(object):
    """Test sample from the MNIST dataset."""

    def __new__(cls, root):
        return get_dataset(datasets.MNIST, root, train=False)


class KMNIST_Train(object):
    """Train sample from the Kuzushiji-MNIST dataset.
    https://github.com/rois-codh/kmnist
    """

    def __new__(cls, root, train_size=None, random_state=None):
        return get_train_dataset(datasets.KMNIST, root, train_size=train_size,
                                 random_state=random_state)


class KMNIST_Test(object):
    """Test sample from the Kuzushiji-MNIST dataset."""

    def __new__(cls, root):
        return get_dataset(datasets.KMNIST, root, train=False)


class FashionMNIST_Train(object):
    """Train sample from the Fashion-MNIST dataset."""

    def __new__(cls, root, train_size=None, random_state=None):
        return get_train_dataset(datasets.FashionMNIST, root, train_size=train_size,
                                 random_state=random_state)


class FashionMNIST_Test(object):
    """Test sample from the Fashion-MNIST dataset."""

    def __new__(cls, root):
        return get_dataset(datasets.FashionMNIST, root, train=False)
