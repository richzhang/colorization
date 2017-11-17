import numpy as np
import scipy as sp
import cv2
import matplotlib.pyplot as plt
from skimage import color
import caffe
from sklearn.cluster import KMeans
from skimage.io import imread
from skimage.io import imsave
from skimage import color
import os
import sys
import ntpath
import datetime
from scipy.ndimage.interpolation import zoom


def create_temp_directory(path_template, N=1e8):
    print(path_template)
    cur_path = path_template % np.random.randint(0, N)
    while(os.path.exists(cur_path)):
        cur_path = path_template % np.random.randint(0, N)
    print('Creating directory: %s' % cur_path)
    os.mkdir(cur_path)
    return cur_path


def lab2rgb_transpose(img_l, img_ab):
    ''' INPUTS
            img_l     1xXxX     [0,100]
            img_ab     2xXxX     [-100,100]
        OUTPUTS
            returned value is XxXx3 '''
    pred_lab = np.concatenate((img_l, img_ab), axis=0).transpose((1, 2, 0))
    pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1)*255).astype('uint8')
    return pred_rgb


def rgb2lab_transpose(img_rgb):
    ''' INPUTS
            img_rgb XxXx3
        OUTPUTS
            returned value is 3xXxX '''
    return color.rgb2lab(img_rgb).transpose((2, 0, 1))


class ColorizeImageBase():
    def __init__(self, Xd=256, Xfullres_max=10000):
        self.Xd = Xd
        self.img_l_set = False
        self.net_set = False
        self.Xfullres_max = Xfullres_max  # maximum size of maximum dimension
        self.img_just_set = False  # this will be true whenever image is just loaded
        # net_forward can set this to False if they want

    def prep_net(self):
        raise Exception("Should be implemented by base class")

    # ***** Image prepping *****
    def load_image(self, input_path):
        # rgb image [CxXdxXd]
        im = cv2.cvtColor(cv2.imread(input_path, 1), cv2.COLOR_BGR2RGB)
        self.img_rgb_fullres = im.copy()
        self._set_img_lab_fullres_()

        im = cv2.resize(im, (self.Xd, self.Xd))
        self.img_rgb = im.copy()
        # self.img_rgb = sp.misc.imresize(plt.imread(input_path),(self.Xd,self.Xd)).transpose((2,0,1))

        self.img_l_set = True

        # convert into lab space
        self._set_img_lab_()
        self._set_img_lab_mc_()

    def set_image(self, input_image):
        self.img_rgb_fullres = input_image.copy()
        self._set_img_lab_fullres_()

        self.img_l_set = True

        im = cv2.resize(self.img_rgb_fullres, (self.Xd, self.Xd))
        self.img_rgb = input_image
        # convert into lab space
        self._set_img_lab_()
        self._set_img_lab_mc_()

    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab         2xXxX     input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        if(not self.img_l_set):
            print('I need to have an image!')
            return -1
        if(not self.net_set):
            print('I need to have a net!')
            return -1

        self.input_ab = input_ab
        self.input_ab_mc = (input_ab-self.ab_mean)/self.ab_norm
        self.input_mask = input_mask
        self.input_mask_mult = input_mask*self.mask_mult
        return 0

    def get_result_PSNR(self, result=-1, return_SE_map=False):
        if np.array((result)).flatten()[0] == -1:
            cur_result = self.get_img_forward()
        else:
            cur_result = result.copy()
        SE_map = (1.*self.img_rgb-cur_result)**2
        cur_MSE = np.mean(SE_map)
        cur_PSNR = 20*np.log10(255./np.sqrt(cur_MSE))
        if return_SE_map:
            return(cur_PSNR, SE_map)
        else:
            return cur_PSNR

    def get_img_forward(self):
        # get image with point estimate
        return self.output_rgb

    def get_img_gray(self):
        # Get black and white image
        return lab2rgb_transpose(self.img_l, np.zeros((2, self.Xd, self.Xd)))

    def get_img_gray_fullres(self):
        # Get black and white image
        return lab2rgb_transpose(self.img_l_fullres, np.zeros((2, self.img_l_fullres.shape[1], self.img_l_fullres.shape[2])))

    def get_img_fullres(self):
        # This assumes self.img_l_fullres, self.output_ab are set.
        # Typically, this means that set_image() and net_forward()
        # have been called.
        # bilinear upsample
        zoom_factor = (1, 1.*self.img_l_fullres.shape[1]/self.output_ab.shape[1], 1.*self.img_l_fullres.shape[2]/self.output_ab.shape[2])
        output_ab_fullres = zoom(self.output_ab, zoom_factor, order=1)

        return lab2rgb_transpose(self.img_l_fullres, output_ab_fullres)

    def get_input_img_fullres(self):
        zoom_factor = (1, 1.*self.img_l_fullres.shape[1]/self.input_ab.shape[1], 1.*self.img_l_fullres.shape[2]/self.input_ab.shape[2])
        input_ab_fullres = zoom(self.input_ab, zoom_factor, order=1)
        return lab2rgb_transpose(self.img_l_fullres, input_ab_fullres)

    def get_input_img(self):
        return lab2rgb_transpose(self.img_l, self.input_ab)

    def get_img_mask(self):
        # Get black and white image
        return lab2rgb_transpose(100.*(1-self.input_mask), np.zeros((2, self.Xd, self.Xd)))

    def get_img_mask_fullres(self):
        # Get black and white image
        zoom_factor = (1, 1.*self.img_l_fullres.shape[1]/self.input_ab.shape[1], 1.*self.img_l_fullres.shape[2]/self.input_ab.shape[2])
        input_mask_fullres = zoom(self.input_mask, zoom_factor, order=0)
        return lab2rgb_transpose(100.*(1-input_mask_fullres), np.zeros((2, input_mask_fullres.shape[1], input_mask_fullres.shape[2])))

    def get_sup_img(self):
        return lab2rgb_transpose(50*self.input_mask, self.input_ab)

    def get_sup_fullres(self):
        zoom_factor = (1, 1.*self.img_l_fullres.shape[1]/self.output_ab.shape[1], 1.*self.img_l_fullres.shape[2]/self.output_ab.shape[2])
        input_mask_fullres = zoom(self.input_mask, zoom_factor, order=0)
        input_ab_fullres = zoom(self.input_ab, zoom_factor, order=0)
        return lab2rgb_transpose(50*input_mask_fullres, input_ab_fullres)

    # ***** Private functions *****
    def _set_img_lab_fullres_(self):
        # adjust full resolution image to be within maximum dimension is within Xfullres_max
        Xfullres = self.img_rgb_fullres.shape[0]
        Yfullres = self.img_rgb_fullres.shape[1]
        if Xfullres > self.Xfullres_max or Yfullres > self.Xfullres_max:
            if Xfullres > Yfullres:
                zoom_factor = 1.*self.Xfullres_max/Xfullres
            else:
                zoom_factor = 1.*self.Xfullres_max/Yfullres
            self.img_rgb_fullres = zoom(self.img_rgb_fullres, (zoom_factor, zoom_factor, 1), order=1)

        self.img_lab_fullres = color.rgb2lab(self.img_rgb_fullres).transpose((2, 0, 1))
        self.img_l_fullres = self.img_lab_fullres[[0], :, :]
        self.img_ab_fullres = self.img_lab_fullres[1:, :, :]

    def _set_img_lab_(self):
        # set self.img_lab from self.im_rgb
        self.img_lab = color.rgb2lab(self.img_rgb).transpose((2, 0, 1))
        self.img_l = self.img_lab[[0], :, :]
        self.img_ab = self.img_lab[1:, :, :]

    def _set_img_lab_mc_(self):
        # set self.img_lab_mc from self.img_lab
        # lab image, mean centered [XxYxX]
        self.img_lab_mc = self.img_lab / np.array((self.l_norm, self.ab_norm, self.ab_norm))[:, np.newaxis, np.newaxis]-np.array((self.l_mean/self.l_norm, self.ab_mean/self.ab_norm, self.ab_mean/self.ab_norm))[:, np.newaxis, np.newaxis]
        self._set_img_l_()

    def _set_img_l_(self):
        self.img_l_mc = self.img_lab_mc[[0], :, :]
        self.img_l_set = True

    def _set_img_ab_(self):
        self.img_ab_mc = self.img_lab_mc[[1, 2], :, :]

    def _set_out_ab_(self):
        self.output_lab = rgb2lab_transpose(self.output_rgb)
        self.output_ab = self.output_lab[1:, :, :]


class ColorizeImageCaffe(ColorizeImageBase):
    def __init__(self, Xd=256):
        print('ColorizeImageCaffe instantiated')
        ColorizeImageBase.__init__(self, Xd)
        self.l_norm = 1.
        self.ab_norm = 1.
        self.l_mean = 50.
        self.ab_mean = 0.
        self.mask_mult = 110.

        self.pred_ab_layer = 'pred_ab'  # predicted ab layer

        # Load grid properties
        self.pts_in_hull_path = './data/color_bins/pts_in_hull.npy'
        self.pts_in_hull = np.load(self.pts_in_hull_path)  # 313x2, in-gamut

    # ***** Net preparation *****
    def prep_net(self, gpu_id, prototxt_path='', caffemodel_path=''):
        print('gpu_id = %d, net_path = %s, model_path = %s' % (gpu_id, prototxt_path, caffemodel_path))
        if gpu_id == -1:
            caffe.set_mode_cpu()
        else:
            caffe.set_device(gpu_id)
            caffe.set_mode_gpu()
        self.gpu_id = gpu_id
        self.net = caffe.Net(prototxt_path, caffemodel_path, caffe.TEST)
        self.net_set = True

        # automatically set cluster centers
        if len(self.net.params[self.pred_ab_layer][0].data[...].shape) == 4 and self.net.params[self.pred_ab_layer][0].data[...].shape[1] == 313:
            print('Setting ab cluster centers in layer: %s' % self.pred_ab_layer)
            self.net.params[self.pred_ab_layer][0].data[:, :, 0, 0] = self.pts_in_hull.T

        # automatically set upsampling kernel
        for layer in self.net._layer_names:
            if layer[-3:] == '_us':
                print('Setting upsampling layer kernel: %s' % layer)
                self.net.params[layer][0].data[:, 0, :, :] = np.array(((.25, .5, .25, 0), (.5, 1., .5, 0), (.25, .5, .25, 0), (0, 0, 0, 0)))[np.newaxis, :, :]

    # ***** Call forward *****
    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab         2xXxX     input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        if ColorizeImageBase.net_forward(self, input_ab, input_mask) == -1:
            return -1

        net_input_prepped = np.concatenate((self.img_l_mc, self.input_ab_mc, self.input_mask_mult), axis=0)

        self.net.blobs['data_l_ab_mask'].data[...] = net_input_prepped
        self.net.forward()

        # return prediction
        self.output_rgb = lab2rgb_transpose(self.img_l, self.net.blobs[self.pred_ab_layer].data[0, :, :, :])

        self._set_out_ab_()
        return self.output_rgb

    def get_img_forward(self):
        # get image with point estimate
        return self.output_rgb

    def get_img_gray(self):
        # Get black and white image
        return lab2rgb_transpose(self.img_l, np.zeros((2, self.Xd, self.Xd)))


class ColorizeImageCaffeGlobDist(ColorizeImageCaffe):
    # Caffe colorization, with additional global histogram as input
    def __init__(self, Xd=256):
        ColorizeImageCaffe.__init__(self, Xd)
        self.glob_mask_mult = 1.
        self.glob_layer = 'glob_ab_313_mask'

    def net_forward(self, input_ab, input_mask, glob_dist=-1):
        # glob_dist is 313 array, or -1
        if np.array(glob_dist).flatten()[0] == -1:  # run without this, zero it out
            self.net.blobs[self.glob_layer].data[0, :-1, 0, 0] = 0.
            self.net.blobs[self.glob_layer].data[0, -1, 0, 0] = 0.
        else:  # run conditioned on global histogram
            self.net.blobs[self.glob_layer].data[0, :-1, 0, 0] = glob_dist
            self.net.blobs[self.glob_layer].data[0, -1, 0, 0] = self.glob_mask_mult

        self.output_rgb = ColorizeImageCaffe.net_forward(self, input_ab, input_mask)
        self._set_out_ab_()
        return self.output_rgb


class ColorizeImageCaffeDist(ColorizeImageCaffe):
    # caffe model which includes distribution prediction
    def __init__(self, Xd=256):
        ColorizeImageCaffe.__init__(self, Xd)
        self.dist_ab_set = False
        self.scale_S_layer = 'scale_S'
        self.dist_ab_S_layer = 'dist_ab_S'  # softened distribution layer
        self.pts_grid = np.load('./data/color_bins/pts_grid.npy')  # 529x2, all points
        self.in_hull = np.load('./data/color_bins/in_hull.npy')  # 529 bool
        self.AB = self.pts_grid.shape[0]  # 529
        self.A = int(np.sqrt(self.AB))  # 23
        self.B = int(np.sqrt(self.AB))  # 23
        self.dist_ab_full = np.zeros((self.AB, self.Xd, self.Xd))
        self.dist_ab_grid = np.zeros((self.A, self.B, self.Xd, self.Xd))
        self.dist_entropy = np.zeros((self.Xd, self.Xd))

    def prep_net(self, gpu_id, prototxt_path='', caffemodel_path='', S=.2):
        ColorizeImageCaffe.prep_net(self, gpu_id, prototxt_path=prototxt_path, caffemodel_path=caffemodel_path)
        self.S = S
        self.net.params[self.scale_S_layer][0].data[...] = S

    def net_forward(self, input_ab, input_mask):
        # INPUTS
        #     ab         2xXxX     input color patches (non-normalized)
        #     mask     1xXxX    input mask, indicating which points have been provided
        # assumes self.img_l_mc has been set

        function_return = ColorizeImageCaffe.net_forward(self, input_ab, input_mask)
        if np.array(function_return).flatten()[0] == -1:  # errored out
            return -1

        # set distribution
        # in-gamut, CxXxX, C = 313
        self.dist_ab = self.net.blobs[self.dist_ab_S_layer].data[0, :, :, :]
        self.dist_ab_set = True

        # full grid, ABxXxX, AB = 529
        self.dist_ab_full[self.in_hull, :, :] = self.dist_ab

        # gridded, AxBxXxX, A = 23
        self.dist_ab_grid = self.dist_ab_full.reshape((self.A, self.B, self.Xd, self.Xd))

        # return
        return function_return

    def get_ab_reccs(self, h, w, K=5, N=25000, return_conf=False):
        ''' Recommended colors at point (h,w)
        Call this after calling net_forward
        '''
        if not self.dist_ab_set:
            print('Need to set prediction first')
            return 0

        # randomly sample from pdf
        cmf = np.cumsum(self.dist_ab[:, h, w])  # CMF
        cmf = cmf/cmf[-1]
        cmf_bins = cmf

        # randomly sample N points
        rnd_pts = np.random.uniform(low=0, high=1.0, size=N)
        inds = np.digitize(rnd_pts, bins=cmf_bins)
        rnd_pts_ab = self.pts_in_hull[inds, :]

        # run k-means
        kmeans = KMeans(n_clusters=K).fit(rnd_pts_ab)

        # sort by cluster occupancy
        k_label_cnt = np.histogram(kmeans.labels_, np.arange(0, K+1))[0]
        k_inds = np.argsort(k_label_cnt, axis=0)[::-1]

        cluster_per = 1. * k_label_cnt[k_inds]/N  # percentage of points within cluster
        cluster_centers = kmeans.cluster_centers_[k_inds, :]  # cluster centers

        # cluster_centers = np.random.uniform(low=-100,high=100,size=(N,2))
        if return_conf:
            return cluster_centers, cluster_per
        else:
            return cluster_centers

    def compute_entropy(self):
        # compute the distribution entropy (really slow right now)
        self.dist_entropy = np.sum(self.dist_ab*np.log(self.dist_ab), axis=0)

    def plot_dist_grid(self, h, w):
        # Plots distribution at a given point
        plt.figure()
        plt.imshow(self.dist_ab_grid[:, :, h, w], extent=[-110, 110, 110, -110], interpolation='nearest')
        plt.colorbar()
        plt.ylabel('a')
        plt.xlabel('b')

    def plot_dist_entropy(self):
        # Plots distribution at a given point
        plt.figure()
        plt.imshow(-self.dist_entropy, interpolation='nearest')
        plt.colorbar()
