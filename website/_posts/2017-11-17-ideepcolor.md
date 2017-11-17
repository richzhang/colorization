---
layout: post
title: Real-Time User-Guided Image Colorization with Learned Deep Priors
description: we produce a user-guided system for image colorization, allowing for quick exploration of possible image colorizations
img: ideepcolor.jpg
color: d14568
---

{{page.description}}

We propose a deep learning approach for user-guided image colorization. The system directly maps a grayscale image, along with sparse, local user ``hints" to an output colorization with a Convolutional Neural Network (CNN). Rather than using hand-defined rules, the network propagates user edits by fusing low-level cues along with high-level semantic information, learned from large-scale data. We train on a million images, with simulated user inputs. To guide the user towards efficient input selection, the system recommends likely colors based on the input image and current user inputs. The colorization is performed in a single feed-forward pass, enabling real-time use. Even with randomly simulated user inputs, we show that the proposed system helps novice users quickly create realistic colorizations, and show large improvements in colorization quality with just a minute of use. In addition, we show that the framework can incorporate other user "hints" as to the desired colorization, showing an application to color histogram transfer. 

Richard Zhang, Jun-Yan Zhu, Phillip Isola, Xinyang Geng, Angela S. Lin, Tianhe Yu, Alexei A. Efros.

[https://arxiv.org/abs/1705.02999](https://arxiv.org/abs/1705.02999)

## Dataset

{% highlight md %}

interactive-deep-colorization

{% endhighlight %}
