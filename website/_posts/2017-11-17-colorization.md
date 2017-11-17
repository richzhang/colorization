---
layout: post
title: Colorful Image Colorization
description: learning to colorize images is a way to learn about the natural image statistics
img: colorization.jpg
color: 078258
---

{{page.description}}

Given a grayscale photograph as input, this paper attacks the problem of hallucinating a plausible color version of the photograph.	This problem is clearly underconstrained, so previous approaches have either relied on significant user interaction or resulted in desaturated colorizations. We propose a fully automatic approach that produces vibrant and realistic colorizations. We embrace the underlying uncertainty of the	problem by posing it as a classification task and use class-rebalancing at training time to increase the diversity of colors in the result. The system is implemented as a feed-forward pass in a CNN at test time and is trained on over a million color images. We evaluate our algorithm using a "colorization Turing test," asking human participants to choose between a generated and ground truth color image. Our method successfully fools	humans on 32% of the trials, significantly higher than previous methods. Moreover, we show that colorization can be a powerful pretext task for	self-supervised feature learning, acting as a cross-channel encoder. This approach results in state-of-the-art performance on several feature learning benchmarks. 

Richard Zhang, Phillip Isola, Alexei A. Efros.

[https://arxiv.org/pdf/1603.08511.pdf](https://arxiv.org/pdf/1603.08511.pdf)

## Dataset

{% highlight md %}

colorization

{% endhighlight %}
