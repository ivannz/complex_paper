# Structure

This folder contains experiments on MusicNet dataset.

## Subfolders

### ./data
Containts the HDF5 for the MusicNet dataset original 44kHz and downsampled 11kHz.
Train, validation and test samples are built using `../extra/get_datasets.ipynb`.

### ./runs
This folder houses the manifests and results of the grid search experiments in its subfolders.

### ./legacy
Contains experiments and processing notebooks for the very first major run of the grid search, in which i tried to replicate the results of Trabelsi et al. (2017), and see how well the models can be compressed. The major flaw was the layer averaged KL divergence, which effectively resulted in more uniform compression rate among layers. This was a detriment to the resulting compression, since in general a good policy compresses layers nonuniformly, as was pointed out by He et al. (2018). Intutitively, the upstream layers are closer to the input data and thus should be less sparse, whilst downstream layers, which deal mostly with feature generalisation in contrast may be sparse.

## Notebooks

### auto_experiment.ipynb
A notebook for setting up an experiment and running it using `.auto`.

### experiment_analysis.ipynb
Collect performace scores from each experiment in a specified folder, reconstruct the parameter grid, aggregate the scores and make a plot.

### experiment_generator.ipynb
A notebook to generate manifests for the experiment with variational complex-valued network compression on MusicNet 11kHz.

Experiment setup (replicated x5)
* `Experiment-1` coarse grid
    * 50-75-50 with optim-restart on fine tuning
    * early stopping 20/0 (patience/cooldown) with 0% + 0.02
        - disabled during compression stage
    * KL-div penalty `summing` with base coefficient 1e-5
    * shifted complex FFT features
    * penalty coefficients {1.0, 2.5, 5.0, 7.5} * [1e-5, 1e-4, 1e-3, 1e-2] + [1.0]
* `Experiment-2` finer grid around the peak of the validation scores
    * same as Experiment-1
    * penalty coefficients {2.5, 3.75, 6.75, 8.75} * [1e-3, 1e-2] (excluding values from Experiment-1)
