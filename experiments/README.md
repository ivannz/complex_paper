# Experiments

This folder contains the notebooks for replicating the figures in the paper and its appendix. There are three types of notebooks here:
* `auto_experiment` to run one-shot experiments and tinker with parameters, scheduler settings, and architectures etc.
* `experiment_generator` to specify the parameter grid and create experiment manifests in bulk in a specified folder, which is later run using  with `cplxpaper.auto`
* `figure__` to plot figures based on the surrogate data of the reports built from successful experiments using `cplxpaper.auto.reports`

The basic pipeline for an experiment is as follows:
1. create a manifest of an experiment (json or dict)
2. run by calling `cplxpaper.auto.run(...)`
3. use either a model viewer from `extra`, or build a report/figure form the successful run.
