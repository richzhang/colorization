# Data-dependent initialization of convolutional neural networks

Created by Philipp Krähenbühl.

### Introduction

This code implements the initialization presented in our [arXiv tech report](http://arxiv.org/abs/1511.06856), which is under submission at ICLR 2016.

*This is a reimplementation and currently work in progress. Use at your own risk.*

### License

This code is released under the BSD License (refer to the LICENSE file for details).

### Citing

If you find our initialization useful in your research, please consider citing:

    @article{krahenbuhl2015data,
      title={Data-dependent Initializations of Convolutional Neural Networks},
      author={Kr{\"a}henb{\"u}hl, Philipp and Doersch, Carl and Donahue, Jeff and Darrell, Trevor},
      journal={arXiv preprint arXiv:1511.06856},
      year={2015}
    }

### Setup

Checkout the project and create a symlink to caffe in the `magic_init` directory:
```Shell
ln -s path/to/caffe/python/caffe caffe
```

### Examples

Here is a quick example on how to initialize alexnet:
```bash
python magic_init.py path/to/alexnet/deploy.prototxt path/to/output.caffemodel -d "path/to/some/images/*.png" -q -nit 10 -cs
```
Here ```-d``` flag allows you to initialize the network using your own images. Feel free to use imagenet, Pascal, COCO or whatever you have at hand, it shouldn't make a big difference. The ```-q``` (queit) flag suppresses all the caffe logging, ```-nit``` controls the number of batches used (while ```-bs``` controls the batch size). Finally ```-cs``` rescales the gradients accross layers. This rescaling currently works best for feed-forward networks, and might not work too well for DAG structured networks (we are working on that).

To run the k-means initialization use:
```bash
python magic_init.py path/to/alexnet/deploy.prototxt path/to/output.caffemodel -d "path/to/some/images/*.png" -q -nit 10 -cs -t kmeans
```

Finally, ```python magic_init.py -h``` should provide you with more help.


### Pro tips
If you're numpy implementation is based on openblas, try disabeling threading ```export OPENBLAS_NUM_THREADS=1```, it can improve the runtime performance a bit.
