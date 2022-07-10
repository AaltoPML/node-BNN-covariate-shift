---
title: Tackling covariate shift with node-based Bayesian neural networks
publication: Oral presentation at the International Conference on Machine Learning (ICML) 2022
description: Trung Trinh, Markus Heinonen, Luigi Acerbi, Samuel Kaski
---

*This website contains information regarding the paper Tackling covariate shift with node-based Bayesian neural networks.*

Node-based Bayesian neural networks (node-BNNs) have demonstrated good generalization under input corruptions.

In this work, we provide insights into the robustness of node-BNNs under corruptions and propose a simple method to further improve this robustness.

## Node-based Bayesian neural networks

The standard Bayesian treatment of neural networks is to place a prior distribution \\(p(\theta)\\) over the parameters \\(\theta\\) (weights and biases) and infer their posterior distribution  given the training data \\(\mathcal{D}\\) using Bayes' rule.

$$p(\theta | \mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$$

The resulting model is aptly named *Bayesian neural networks (BNNs)*.
