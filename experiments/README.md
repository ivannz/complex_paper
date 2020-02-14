# Experiments


## MusicNet

Classical music annotation (Thickstun et. al, 2017). 330 musical compositions from the public
domain. Waveforms are sampled at 44.1 kHz and are aligned with overlapping `[begin, end)`
intervals each annotated with instrument, muscal note label, musical measure, beat and note
value. There are 83 distinct note labels 21, 23-104, i.e. 22 is missing.


### Trabelsi et al. (2019)

MusciNet contains 330 compositions: 6 for validation, 3 for test and 321 for training.
```python
{
    "test": [
        "id_2303", "id_2382", "id_1819",
    ],
    "valid": [
        "id_2131", "id_2384", "id_1792", "id_2514", "id_2567", "id_1876",
    ]
}
```

Trabelsi et al. (2019) use the dataset for multilabel classification with complex-valued
neural networks. They compare real against cplx, shallow against deep, and raw (time-domain)
against FFT (spectral domain).

#### Specifics

They downsample 44.1kHz down to 11kHz and make a sliding window of 4096 samples with stride
one, i.e. approx 0.4 sec. of audio every 100 micro seconds. A batch is formed by taking **every
composition** from `train` and then picking a random window of the waveform in each one (code).
The musical note labels are taken at the mid-point of the sampled window.

They train every model in each case for **200** epoch, where one epoch is **1000** batches
(code). Adam is used with LR schedule (code):
* 1e-3{<10}, 1e-4{<100}, 5e-5{<120}, 1e-5{<150}, 1e-6{>=150}.

Batches are propocessed differently, depending on the model type. For a waveform window
$x_t = (x_{ti})_{i=1}^w \in \mathbb{R}^w$:
* (raw) just returns the input
* (fourier) apply FFT to $x_t$ to get $z_t \in \mathbb{C}^w$
* (STFT) perform batched Short Time Fourier Transform with windows of size **l=120**
overlaping by **o=60** to get $z_t \in \mathbb{C}^{\tfrac{l+2}2 \times \tfrac{w+l-o}{l-o}}$,
where the first dimension of $z_t$ are frequencies and the last is the window times.
**NOTE** they treat times as features and frequencies as sequences
[dataset.py#L179](https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/musicnet/musicnet/dataset.py#L179)
(see `check_original_transforms.ipynb`)

Inputs in different models
* (FFT, STFT)
  * ($\mathbb{C}^w$): $z_t$
  * ($\mathbb{R}^w$): $\lvert z_t \rvert = (\lvert z_{ti} \rvert)\_{i=1}^w$
* (raw)
  * ($\mathbb{C}^w$): $z_t = x_t$ (train.py#L190 forces one trailing channel!)
  * ($\mathbb{R}^w$): $x_t$

Since the dataset is severely imbalanced in each musical note, the performance is measured using
the area under precision-recall curve, or average precision score.
On [callbacks.py#L90](https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/musicnet/musicnet/callbacks.py#L90)
the scores and targets are flattened, so the AP-score is computed as if every note is the same class.
(**NOTE** can this approach misrepresent tthe perfromance?)

No explicit early stopping is in their code, but they do checkpointing every epoch.

Validation adn test sets are strided with 128 (`construct_eval_set()` in [dataset.py#L117](https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/musicnet/musicnet/dataset.py#L117)
called from `eval_set()` at [dataset.py#L210](https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/musicnet/musicnet/dataset.py#L210))


### Yang et al. (2019)

Propose complex-valued attention layer and develop complex-domain transformer.

#### Specifics

AP-score is measured via `.flatten()`
[train.py#L103](https://github.com/muqiaoy/dl_signal/blob/master/transformer/train.py#L103)
By default they clip grads to `0.35`. Batch size `16` and `2k` epochs.

Use fft features with complex-valued output. Use differrent strides for training (512) and
testing (128) [parse_file.py#L34](https://github.com/muqiaoy/dl_signal/blob/master/music/parse_file.py#L34).


### Current Paper

Makes heavy use of [cplxmodule](https://github.com/ivannz/cplxmodule.git).

#### Planned experiments

We use MusicNet (Thickstun et al., 2017) with preprocessin as in (Trabelsi et al., 2019).
Motivated by the experiments in (Gale et al., 2019) we test complex-domain compression on
transformer and deep convolutional network architectures in application to the same task.
We expect differing sparsity patterns between these architectures.

##### Experiemntal setup

#### Components

Real networks ordinary network in real domain
* (existing) **torch.nn**: BatchNorm1d; **cplxmodule.nn.relevance**: Linear{ARD, Masked};
* (~missing~ fixed) **cplxmodule.nn.relevance**: Conv1d{ARD, Masked}, Conv2d{ARD, Masked} (see `mlss2019`)

Complex-valued networks uxing `Cplx` form `cplxmodule`
* (existing) **cplxmodule.nn.layers**: CplxConv1d, CplxLinear, RealToCplx, AsTypeCplx, CplxToReal;
  **cplxmodule.nn.relevance**: CplxLinear{ARD, Masked}
* (~missing~ fixed) **cplxmodule.nn.relevance**: CplxConv1d{ARD, Masked},  CplxConv2d{ARD, Masked}
* (~missing~ fixed) **cplxmodule.nn.layers**: CplxBatchNorm1d


* raw signal: 

experiment plan:
components:
* train (model, criterion, feed, verbose, kl-div)