<!--<h3><b>Colorful Image Colorization</b></h3>-->
## <b>Colorful Image Colorization</b> [[Project Page]](http://richzhang.github.io/colorization/) <br>
[Richard Zhang](https://richzhang.github.io/), [Phillip Isola](http://web.mit.edu/phillipi/), [Alexei A. Efros](http://www.eecs.berkeley.edu/~efros/). In [ECCV, 2016](http://arxiv.org/pdf/1603.08511.pdf).

![Teaser Image](http://richzhang.github.io/colorization/resources/images/teaser4.jpg)

### Note ###

**Sept 2020** The original implementation is in the master branch and was in Caffe. Since it has been 3-4 years, I decided to make a simple PyTorch test-time script. I also added our interactive method from SIGGRAPH 2017 (which can also do automatic colorization).

### Quick and easy start

```
git clone https://github.com/richzhang/colorization.git
cd colorization
git checkout pytorch
pip install requirements.txt
python demo_release.py # colorization a demo image
```

See [demo_release.py](demo_release.py) for how to run the model, since there are some pre and post-processing steps.

```python
import colorizers
colorizer_eccv16 = colorizers.eccv16(pretrained=True).eval()
colorizer_siggraph17 = colorizers.siggraph17(pretrained=True).eval()
```

### Original implementation functionalities

The original implementation contained the following. It is in Caffe and is no longer supported. Please see the **master** branch for it.

<b>Colorization-centric functionality</b>
 - (0) a test time script to colorize an image (python script)
 - (1) a test time demonstration (IPython Notebook)
 - (2) code for training a colorization network
 - (3) links to our results on the ImageNet test set, along with a pointer to AMT real vs fake test code

<b>Representation Learning-centric functionality</b>
 - (4) pre-trained AlexNet, used for representation learning tests (Section 3.2)
 - (5) code for training AlexNet with colorization
 - (6) representation learning tests


### Citation ###

If you find this model useful for your resesarch, please cite with these bibtexs for the [ECCV 2016](http://richzhang.github.io/colorization/resources/bibtex_eccv2016_colorization.txt) and [SIGGRAPH 2017](http://richzhang.github.io/colorization/resources/bibtex_siggraph2017.txt) papers

### Misc ###
Contact Richard Zhang at rich.zhang at eecs.berkeley.edu for any questions or comments.
