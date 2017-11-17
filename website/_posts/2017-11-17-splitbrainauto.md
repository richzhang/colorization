---
layout: post
title: Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction
description: cross-channel prediction achieves high unsupervised/self-supervised learning performance
img: splitbrainauto.png
color: 509ace
---

{{page.description}}

We propose split-brain autoencoders, a straightforward modification of the traditional autoencoder architecture, for unsupervised representation learning. The method adds a split to the network, resulting in two disjoint sub-networks. Each sub-network is trained to perform a difficult task -- predicting one subset of the data channels from another. Together, the sub-networks extract features from the entire input signal. By forcing the network to solve cross-channel prediction tasks, we induce a representation within the network which transfers well to other, unseen tasks. This method achieves state-of-the-art performance on several large-scale transfer learning benchmarks.

Richard Zhang, Phillip Isola, Alexei A. Efros.

[https://arxiv.org/abs/1611.09842](https://arxiv.org/abs/1611.09842)

## Dataset

{% highlight md %}

splitbrainauto

{% endhighlight %}
