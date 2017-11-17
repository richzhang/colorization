
import numpy as np
import warnings
import os
import sklearn.neighbors as nn
import caffe
from skimage import color
import matplotlib.pyplot as plt
import color_quantization as cq

# ***************************************
# ***** LAYERS FOR GLOBAL HISTOGRAM *****
# ***************************************
class SpatialRepLayer(caffe.Layer):
    '''
    INPUTS
        bottom[0].data  NxCx1x1
        bottom[1].data  NxCxXxY
    OUTPUTS
        top[0].data     NxCxXxY     repeat 0th input spatially  '''
    def setup(self,bottom,top):
        if(len(bottom)!=2):
            raise Exception("Layer needs 2 inputs")

        self.param_str_split = self.param_str.split(' ')
        # self.keep_ratio = float(self.param_str_split[0]) # frequency keep whole input

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

        if(self.X!=1 or self.Y!=1):
            raise Exception("bottom[0] should have spatial dimensions 1x1")

        # self.Nref = bottom[1].data.shape[0]
        # self.Cref = bottom[1].data.shape[1]
        self.Xref = bottom[1].data.shape[2]
        self.Yref = bottom[1].data.shape[3]

    def reshape(self,bottom,top):
        top[0].reshape(self.N,self.C,self.Xref,self.Yref) # output shape

    def forward(self,bottom,top):
        top[0].data[...] = bottom[0].data[:,:,:,:] # will do singleton expansion

    def backward(self,top,propagate_down,bottom):
        bottom[0].diff[:,:,0,0] = np.sum(np.sum(top[0].diff,axis=2),axis=2)
        bottom[1].diff[...] = 0


class BGR2HSVLayer(caffe.Layer):
    ''' Layer converts BGR to HSV
    INPUTS    
        bottom[0]   Nx3xXxY     
    OUTPUTS
        top[0].data     Nx3xXxY     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if(len(bottom)!=1):
            raise Exception("Layer should a single input")
        if(bottom[0].data.shape[1]!=3):
            raise Exception("Input should be 3-channel BGR image")

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,3,self.X,self.Y)
 
    def forward(self, bottom, top):
        for nn in range(self.N):
            top[0].data[nn,:,:,:] = color.rgb2hsv(bottom[0].data[nn,::-1,:,:].astype('uint8').transpose((1,2,0))).transpose((2,0,1))

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            # bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class BGR2LabLayer(caffe.Layer):
    ''' Layer converts BGR to Lab
    INPUTS    
        bottom[0]   Nx3xXxY     
    OUTPUTS
        top[0].data     Nx3xXxY     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if(len(bottom)!=1):
            raise Exception("Layer should a single input")
        if(bottom[0].data.shape[1]!=3):
            raise Exception("Input should be 3-channel BGR image")

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,3,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = color.rgb2lab(bottom[0].data[:,::-1,:,:].astype('uint8').transpose((2,3,0,1))).transpose((2,3,0,1))

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            # bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class ColorGlobalDropoutLayer(caffe.Layer):
    '''
    Inputs
        bottom[0].data      NxCx1x1
    Outputs
        top[0].data         Nx(C+1)x1x1     last channel is whether or not to keep input
                                            first C channels are copied from bottom (if kept)
    '''
    def setup(self,bottom,top):
        if(len(bottom)==0):
            raise Exception("Layer needs inputs")   

        self.param_str_split = self.param_str.split(' ')
        self.keep_ratio = float(self.param_str_split[0]) # frequency keep whole input
        self.cnt = 0

        self.N = bottom[0].data.shape[0]
        self.C = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self,bottom,top):
        top[0].reshape(self.N,self.C+1,self.X,self.Y) # output mask

    def forward(self,bottom,top):
        top[0].data[...] = 0
        # top[0].data[:,:self.C,:,:] = bottom[0].data[...]

        # determine which ones are kept
        keeps = np.random.binomial(1,self.keep_ratio,size=self.N)

        top[0].data[:,-1,:,:] = keeps[:,np.newaxis,np.newaxis]
        top[0].data[:,:-1,:,:] = bottom[0].data[...]*keeps[:,np.newaxis,np.newaxis,np.newaxis]

    def backward(self,top,propagate_down,bottom):
        0; # backward not implemented


class NNEncLayer(caffe.Layer):
    ''' Layer which encodes ab map into Q colors
    INPUTS    
        bottom[0]   Nx2xXxY     
    OUTPUTS
        top[0].data     NxQ     
    '''
    def setup(self,bottom, top):
        warnings.filterwarnings("ignore")

        if len(bottom) == 0:
            raise Exception("Layer should have inputs")
        # self.NN = 10.
        self.NN = 1.
        self.sigma = 5.
        self.ENC_DIR = './data/color_bins'
        self.nnenc = cq.NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]
        self.Q = self.nnenc.K

    def reshape(self, bottom, top):
        top[0].reshape(self.N,self.Q,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.nnenc.encode_points_mtx_nd(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)
