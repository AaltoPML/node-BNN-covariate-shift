---
title: Tackling covariate shift with node-based Bayesian neural networks
publication: Oral presentation at the International Conference on Machine Learning (ICML) 2022
description: Trung Trinh, Markus Heinonen, Luigi Acerbi, Samuel Kaski
---

*This website contains information regarding the paper Tackling covariate shift with node-based Bayesian neural networks.*

Node-based Bayesian neural networks (node-BNNs) have demonstrated good generalization under input corruptions.

In this work, we provide insights into the robustness of node-BNNs under corruptions and propose a simple method to further improve this robustness.

## Covariate shift due to input corruptions

Generalization is a core problem in machine learning.
The standard set up in supervised learning is to fit a model to a training dataset \\(\mathcal{D}\_{train}\\) and then evaluate its generalization ability on a separate test dataset \\(\mathcal{D}\_{test}\\). Both of these datasets are assumed to contain independent and identically distributed (i.i.d.) samples from the data distribution \\(p(x)\\).

However, performance measured on \\(\mathcal{D}\_{test}\\) only reflects the model's generalization on in-distribution (ID) inputs, i.e., samples coming from \\(p(x)\\). In an open world setting, the model might encounter inputs coming from a different distribution \\(\tilde{p}(x)\\). These inputs are called out-of-distribution (OOD) samples.
The distributional difference between training and test samples is a problem called *covariate shift*.
Currently, neural networks (NNs) have excellent ID performance while behave unpredictably on OOD samples.

In this work, we focus on improving generalization of NNs under *input corruptions*, which is a form of covariate shift.
Input corruptions can happen due to noises or sensor malfunctions.
Some examples of image corruptions are shown below.

## Bayesian neural networks

Bayesian methods are often applied to covariate shift problems.
The standard Bayesian treatment of NNs is to place a prior distribution \\(p(\theta)\\) over the parameters \\(\theta\\) (weights and biases) and infer their posterior distribution \\(p(\theta \| \mathcal{D})\\) given the training data \\(\mathcal{D}\\) using Bayes' rule:

$$p(\theta | \mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$$

The resulting model is aptly named *Bayesian neural networks (BNNs)*.

Due to the large amounts of parameters in a modern NNs, it is computationally expensive to approximate the posterior \\(p(\theta \| \mathcal{D})\\). 
Furthermore, a recent work have showed that BNNs with high fidelity posterior approximations actually perform worse than maximum-a-posteriori (MAP) models under corruptions [cite].

## Node-based Bayesian neural networks

Node-BNNs are recently introduced as an efficient alternative to standard weight-based BNNs.
In a node-BNN, we keep the parameters deterministic while inducing uncertainty over the outputs by multiplying hidden nodes with latent random variables:

\begin{equation}
\begin{aligned}
    \mathbf{f}^{(\ell)} (\mathbf{x}, \mathcal{Z}) = \sigma\left(\mathbf{W}^{(\ell)}\underbrace{(\mathbf{f}^{(\ell-1)} (\mathbf{x}, \mathcal{Z})) \circ \mathbf{z}^{(\ell)})}_{\text{Hadamard product}} + \mathbf{b}^{(\ell)} \right)
\end{aligned}
\end{equation}

### References

