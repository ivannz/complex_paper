from distutils.core import setup, Extension

setup(
    name="cplxpaper",
    version="0.2",
    description="""Package for Bayesian Sparsification """
                """of Complex Valued Networks.""",
    license="MIT License",
    author="Ivan Nazarov",
    author_email="ivan.nazarov@skolkovotech.ru",
    packages=[
        "cplxpaper",
        "cplxpaper.auto",
        "cplxpaper.musicnet",
        "cplxpaper.mnist",
        "cplxpaper.cifar"
    ],
    requires=["torch", "numpy", "pandas", "cplxmodule"],
    # install_requires=[  # uncomment for publishing
    #     "cplxmodule @ git+https://github.com/ivannz/cplxmodule.git"
    # ]
)
