Notes on Compression and Pruning
--------------------------------

There is evidence that $\cplx$-VNN perform better for naturally $\cplx$-valued data
* use half as much storage
* the same number of flops

=> So inducing sparsity becomes important for lower arithmetic complexity, latency, and resource use

Making netowrks smaller or using less ops:
* Knowledge distillation -- creating a smaller replica
* low-rank approximations -- low-rank structure with efficient mat-vec operations
* quantization -- dynamic range of floating-points to fixed-points and use fixed range int arithmetic
* pruning -- remove insignificant parameters introducing sparsity
<!-- * sparsisty -- inducing regularizers (lasso) -->

Related Methods
===============

#### hinton_distilling_2015
Replicate a heavy teacher with light-weight student using high-temperature softmax.
heterogeneous ensembles: a universal model and many specialists for fine grained details with final KL-based distribution optimization

#### balasubramanian_deep_2016
use logit perturbations to simulate the effect of multiple teachers when training a student.

#### lecun_optimal_1990
pruning based on second order information of the loss
saliency [p.~602 OBD recipe~4) is very similar to \frac1\alpha in VarDropout

* HYPOTHESIS: relevance $\propto \frac1\alpha = \frac{\mu^2}{\sigma^2}$ and $\sigma^2 \approx \frac1{\partial^2_{ww} \mathcal{L}}$ at optimum.

#### seide_conversational_2011
$\tau$ threshold-based magnitude pruning (hard ($\ell_0$) or soft ($\ell_1$) thresholding)
with safeguards against reappearance of very weak weights at half the $\tau$

#### zhu_prune_2018
magnitude ranking and adaptive sparsity targeting

#### denton_exploiting_2014
exploit weight redundancy and smoothness to predict weights based on their random subsample
and to get low-rank approximation thereof. Compression - dictionary (rbf or empirical stats)
\+ subsample of weights (factors + loadings)

#### novikov_tensorizing_2015
store weights and convolutions in economical tensor train format, efficient procedures
for computing the mat-vec in TT format

#### courbariaux_training_2015
Post-train fixed point arithmetic conversion or binary quantization during forward and backward

#### chen_fxpnet_2017
Weight-agnostic hashing into groups sharing the same value within a layer: hash->bin->shared
weight. Hashing allows NOT to store indices of the bins

#### han_deep_2016
prune-quantize-compress (zip?) to reduce storage and energy requirements of the nets in
inference mode
* show that pruning with quantization is much better than pruning alone for the final size

1. threshold-based magnitude pruning (train->prune->fine-tune)
2. quantize with $k$-means and fix cluster affinity
3. further fine-tune **shared centroids** using SGD (shared-scatter on forward, grad-sums on backward)

Other notes
-----------
Additive noise parameterization, does not allow easy relevance sharing:
$$
    \alpha_i = \frac{\sigma^2_i}{\lvert \mu_i\rvert^2}
    \,, $$
has to be forced to be close for the grouped weights $w_i \sim(q(w_i)$. It is possible
to use a hyper-prior, like in Louizos et al. (2017) (on the extra multiplicative factor
for the mean $z_g$ and variance $z^2_g$ of each weight c.f. eq. (6)), or introduce a
less Bayesian consensus regularizer:
$$
    \frac{C}2 \sum_{i \in g} \bigl\| \alpha_i - \overline{\alpha}_g \bigr \|^2
    \,, $$
in order to both synchronize in-group relevance and retain the benefit of lower variance
of the stochastic gradient estimator in SGVB.
