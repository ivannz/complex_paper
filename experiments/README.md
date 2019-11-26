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
**NOTE** they treat times as features and frequencies as sequences!

Inputs in different models
* (FFT, STFT)
  * ($\mathbb{C}^w$): $z_t$
  * ($\mathbb{R}^w$): $\lvert z_t \rvert = (\lvert z_{ti} \rvert)\_{i=1}^w$
* (raw)
  * ($\mathbb{C}^w$): $z_t = (1 + 1j) x_t$ (train.py#L190 due to broadcasting)
  * ($\mathbb{R}^w$): $x_t$

### Current Paper

Makes heavy use of [cplxmodule](https://github.com/ivannz/cplxmodule.git).

#### Components

Real networks ordinary network in real domain
* (existing) **torch.nn**: BatchNorm1d; **cplxmodule.relevance**: Linear{ARD, Masked};
* (~missing~ fixed) **cplxmodule.relevance**: Conv1d{ARD, Masked} (see `mlss2019`)

Complex-valued networks uxing `Cplx` form `cplxmodule`
* (existing) **cplxmodule.layers**: CplxConv1d, CplxLinear, RealToCplx, AsTypeCplx, CplxToReal;
  **cplxmodule.relevance**: CplxLinear{ARD, Masked}
* (~missing~ fixed) **cplxmodule.relevance**: CplxConv1d{ARD, Masked}
* (missing) **cplxmodule.layers**: CplxBatchNorm1d


* raw signal: 

experiment plan:
components:
* train (model, criterion, feed, verbose, kl-div)