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
The standard set up in supervised learning is to fit a model to a training dataset \\(\mathcal{D}_{train}\\) and then evaluate its generalization ability on a separate test dataset \\(\mathcal{D}_{test}\\). Both of these datasets are assume to contain independent and identically distributed (i.i.d.) samples from the data distribution \(p(\mathcal{D})\).

## Node-based Bayesian neural networks

The standard Bayesian treatment of neural networks is to place a prior distribution \\(p(\theta)\\) over the parameters \\(\theta\\) (weights and biases) and infer their posterior distribution \\(p(\theta \| \mathcal{D})\\) given the training data \\(\mathcal{D}\\) using Bayes' rule:

$$p(\theta | \mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$$

The resulting model is aptly named *Bayesian neural networks (BNNs)*.

Due to the large amounts of parameters in a modern NNs, it is computationally expensive to approximate the posterior \\(p(\theta \| \mathcal{D})\\). Furthermore, a recent work have showed that BNNs with high fidelity posterior approximations actually perform worse than maximum-a-posteriori (MAP) models under corruptions.

