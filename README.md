<!--<h3><b>Colorful Image Colorization</b></h3>-->
## <b>Colorful Image Colorization</b> [[Project Page]](http://richzhang.github.io/colorization/) <br>
[Richard Zhang](https://richzhang.github.io/), [Phillip Isola](http://web.mit.edu/phillipi/), [Alexei A. Efros](http://www.eecs.berkeley.edu/~efros/). In [ECCV, 2016](http://arxiv.org/pdf/1603.08511.pdf).

![Teaser Image](http://richzhang.github.io/colorization/resources/images/teaser4.jpg)

**Sept 2020 Update** Since it has been 3-4 years, I decided decided to make a minimal PyTorch test-time script and make this the master branch. I also added our interactive method from SIGGRAPH 2017 (which can also do automatic colorization).

### Quick and easy start

Clone the repository and checkout the pytorch branch

```
git clone https://github.com/richzhang/colorization.git
git checkout pytorch
pip install requirements.txt
```

This script will colorize an image. The results should match the images in the `imgs_out` folder.

```
python demo_release.py # colorization a demo image
```

Loading the models is shown below. See [demo_release.py](demo_release.py) for how to run the model, since there are some pre and post-processing steps.

```python
import colorizers
colorizer_eccv16 = colorizers.eccv16().eval()
colorizer_siggraph17 = colorizers.siggraph17().eval()
```

### Original implementation functionality

The original implementation contained train and testing, our network and AlexNet (for representation learning tests), as well as representation learning tests. It is in Caffe and is no longer supported. Please see the **caffe** branch for it.

### Citation ###

If you find this model useful for your resesarch, please cite with these bibtexs for the [ECCV 2016](http://richzhang.github.io/colorization/resources/bibtex_eccv2016_colorization.txt) and [SIGGRAPH 2017](http://richzhang.github.io/colorization/resources/bibtex_siggraph2017.txt) papers

### Misc ###
Contact Richard Zhang at rich.zhang at eecs.berkeley.edu for any questions or comments.
