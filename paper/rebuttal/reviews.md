# Reviews

## Reviewer #1

Borderline paper, but has merits that outweigh flaws.

I am not very familiar with the field of C-valued networks. I have seen talks or skimmed a few papers on this topic, and have not published in this area. I am willing to defend my evaluation, but it is fairly likely that I missed some details, didn't understand some central points, or can't be sure about the novelty of the work.

### Opinion and understanding
This paper extends variational dropout and the local reparametrisation trick to C-valued networks. Applying well known BNN techniques to C-Valued networks can be seen as incremental research. However, it seems to me that the conversion of these methods to the complex domain is non-trivial, making this paper a good contribution.

The authors introduce a 3 step procedure (train deterministic network, sparsify with variational dropout, retrain deterministically) learn sparse models.

### Comments

#### Lack of code
I am concerned about the lack of code provided by the authors. ... C-valued networks seem like somewhat of a niche in the ML communit...

#### Motivation for C-valued networks
It would be nice to include a more detailed statement of the advantages of C networks over regular ones. ... the fundamental motivation for these is not very clear to me from the introduction.

#### Additional Reference
The fine-tuning step that the authors do in the experimental section is motivated in a less heuristic way by [Minimal Random Code Learning: Getting Bits Back from Compressed Model Parameters](https://openreview.net/forum?id=r1f0YiCctm).

#### Structure and exposition
The 3 step training procedure might be better included before the experimental section as it seems like a methodological contribution of the paper.

In the first paragraph of section 2: Although the Bayesian prior over weights can indeed be interpreted as a prior over models, the inclusion of this notation in the first paragraph of section 2 seems unnecessary. Especially considering that the authors refer to weights, and not models, for the remainder of the paper.

In the second paragraph of section 2: “Instead of deriving p(w | D) using the rule …” I think that using the expression “the rule” may be confusing to some readers. You could just say Bayes’ rule.

#### Typos and mistakes
In eq. 2: p(w) should be p(w|D)?

"Plots 2 depict samples from the performance-compression curve on KMNIST and EMNIST- Letters datasets for the studied models and VD compression method (sec. 4.2.1).” Methods is missing an s.

The fonts in the plots are very small and hard to read. It would be nice if they were made a bit bigger.

## Reviewer #2

Missing

## Reviewer #3

Borderline paper, but has merits that outweigh flaws.

I am not very familiar with the field of complex-valued networks, and fairly familiar with Bayesian deep learning and specifically variational dropout. I may update my assessment based on the rebuttal and comments by the other reviewers. I have seen talks or skimmed a few papers on this topic, and have not published in this area. I am willing to defend my evaluation, but it is fairly likely that I missed some details, didn't understand some central points, or can't be sure about the novelty of the work.

### Opinion and understanding
- Derives variational dropout variants for complex-valued networks
- To the best of my knowledge, one of the first applications of variational Bayesian inference to complex-valued networks
- Results may have practical relevance, especially for data with natural complex representations

The paper adapts the variational dropout methods of [1,2,3] to the complex-valued networks [4]. The application is not straightforward but the complications appear to be mostly technical. For this reason I think the paper is borderline.

At the same time, the paper is well-written and well-executed overall, the empirical evaluation is solid. It is also one of the first applications of variational Bayesian inference to complex-valued DNNs. Finally, it appears that the method may have practical relevance. For this reason I am leaning towards acceptance.

### Comments

#### Significance

I think the paper is significant in the following ways:
1. One of the first applications of variational Bayes to complex-valued networks (could the authors please confirm in the rebuttal that this is correct?).
2. One of the first applications of sparsification to decrease computation in complex-valued networks (could the authors please confirm?)
3. Successful results, especially on the music transcription task. Could the authors please comment on how the results compare to state-of-the-art for real-valued networks?

#### Novelty

The paper extends the methodology of [1,2,3] to the complex-valued networks [4]. The methodological novelty is fairly limited, and is in the technical details of the variational and prior distribution forms, and estimation of the KL term in the variational lower bound.

#### Technical Quality

The paper is technically sound.

#### Clarity and Presentation

The paper is clearly written, and easy to follow. I appreciate that the authors included comprehensive background material both on variational dropout and on complex-valued networks.

#### Questions to the authors

In addition to the questions I listed in the significance section:
4. In equation (6’) is rescaling the KL term with C a heuristic or does it have a Bayesian justification?
5. In line 84 you mention that for real-valued networks non-Bayesian methods were found to be preferable for compression. Does this observation translate to complex-valued networks? Would you expect non-Bayesian methods to outperform the proposed method?
6. Related to the previous question, do you expect variational dropout to have any specific advantages over non-Bayesian methods specifically for complex-valued networks?


[1] Variational Dropout and the Local Reparameterization Trick
Diederik P. Kingma, Tim Salimans, Max Welling
[2] Variational Dropout Sparsifies Deep Neural Networks
Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov
[3] Variational Dropout via Empirical Bayes
Valery Kharitonov, Dmitry Molchanov, Dmitry Vetrov
[4] Deep Complex Networks
Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal

## Reviewer #4
Not submitted.

## Reviewer #5

Borderline paper, but the flaws may outweigh the merits.

The paper is well written and the presented approach is sound, however theoretical contributions are modest (direct translations of existing real-valued case methods) and conclusions of the experimental results are not fully clear/drawn (a few more clarifying questions below)

### Opinion and understanding
- Extending popular dropout schemes for real-valued networks to do compression in C-valued networks
- Exploration of the benefits of the suggested schemes in different experimental setups

The paper extends Bayesian dropout methods (specifically, Variational dropout and Automatic Relevance Determination) to C-valued deep networks. It then explores the empirical benefits of these schemes to sparsify the weights of C-valued deep nets on two tasks (image recognition and music transcription).

### Comments

#### Presentation of findings
- It would be very helpful for the reader to summarize the respective strengths/weaknesses of the different schemes to sparsify C-valued networks (VD & ARD): is there one that achieves a better accuracy-compression trade-off (overall or under certain conditions)? Is there one that converges faster than the other ? Is there one that is more stable than the other?

Some of these elements are present in the appendix and could be swapped with some of the less important experimental details

- It would be also very useful to summarize the cases where C-valued networks seem to not deliver performance on par with the R^2 equivalent
(some of these elements are present in the Appendix as well)

#### MNIST experiment

- Based on experimental results, R-valued networks appear to achieve comparable performance to C-valued networks, while being twice smaller. If so, would there be any reason to pick C-valued networks over R-valued networks if the goal is to ultimately compress the network while keeping accuracy high? You point to higher throughput due to fewer floating point multiplications in your conclusion - are there other potential benefits?

- It seems that VD delivers stronger results on that experiment - would you agree?

#### CIFAR10 experiment

- The claim “although the prior used in ARD method offers slightly less compression, it makes up for it by equally slightly better accuracy” is not super clear on the basis of the experimental results
- Performance on CIFAR10 seems somewhat low for a VGG16 network (perhaps not training long enough - this is a minor point)

#### MusicNet experiment

Why is the performance of the uncompressed network lower than results from Trabelsi et al. ? Any chance that the networks have not converged in the low compression regime?

Minor comment: a few typos throughout the paper to be corrected (e.g., m & n indices inverted in equation 10, typo in paragraph 5.5. title)
