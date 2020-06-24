from distutils.core import setup

setup(
    name="cplxpaper",
    version="2020.6",
    description="""Backend for experiments presented in Bayesian"""
                """ Sparsification of Deep Complex-valued Networks.""",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "cplxpaper",
        "cplxpaper.auto",
        "cplxpaper.auto.reports",
        "cplxpaper.musicnet",
        "cplxpaper.musicnet.models",
        "cplxpaper.musicnet.models.real",
        "cplxpaper.musicnet.models.complex",
        "cplxpaper.mnist",
        "cplxpaper.mnist.models",
        "cplxpaper.cifar",
        "cplxpaper.cifar.models",
        "cplxpaper.cifar.models.vgg",
    ],
    install_requires=[
        "torch>=1.4",
        "cplxmodule @ https://github.com/ivannz/cplxmodule/archive/v2020.03.tar.gz",
        "numpy",
        "pandas",
        "sklearn",
        "ncls",
        "resampy",
        "h5py",
    ],
)
