# sparsity
This repository contains some of the sparse neural network experiments conducted during my master's thesis. The current repository only shows random sparse structures with skipped connections and small world networks initialized at the beginning of training.

The experiments are designed to understand the effect of gradient flow for sparse neural networks.

As a side idea, we also looked at parameter factorization formulated as a convolutional operation. This convolutional factorization yields somewhat improved performance compared to pure random sparse structures. The convolutional operation aims to mimic the co-adaptation of neighboring parameters which had been noticed in the literature for sometime. [1]

1. [Predicting Parameters in Deep Learning](https://arxiv.org/abs/1306.0543)
