from distutils.core import setup, Extension

setup(
    name="cplxpaper",
    version="0.5",
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
        "cplxpaper.mnist",
        "cplxpaper.cifar",
        "cplxpaper.musicnet.models",
        "cplxpaper.mnist.models",
        "cplxpaper.cifar.models"
    ],
    requires=[
        "torch",
        "numpy",
        "pandas",
        "cplxmodule",
        "sklearn",
        "ncls",
        "resampy",
        "h5py"
    ],
    # install_requires=[  # uncomment for publishing
    #     "cplxmodule @ git+https://github.com/ivannz/cplxmodule.git"
    # ]
)
