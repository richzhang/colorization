# **************************************
# ***** Richard Zhang / 2016.08.06 *****
# **************************************
import numpy as np
import warnings
import os
import sklearn.neighbors as nn
import caffe
from skimage import color

# ************************
# ***** CAFFE LAYERS *****
# ************************
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
        self.NN = 10.
        self.sigma = 5.
        self.ENC_DIR = './resources/'
        self.nnenc = NNEncode(self.NN,self.sigma,km_filepath=os.path.join(self.ENC_DIR,'pts_in_hull.npy'))

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

class PriorBoostLayer(caffe.Layer):
    ''' Layer boosts ab values based on their rarity
    INPUTS    
        bottom[0]       NxQxXxY     
    OUTPUTS
        top[0].data     Nx1xXxY
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.ENC_DIR = './resources/'
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha,gamma=self.gamma,priorFile=os.path.join(self.ENC_DIR,'prior_probs.npy'))

        self.N = bottom[0].data.shape[0]
        self.Q = bottom[0].data.shape[1]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        top[0].data[...] = self.pc.forward(bottom[0].data[...],axis=1)

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class NonGrayMaskLayer(caffe.Layer):
    ''' Layer outputs a mask based on if the image is grayscale or not
    INPUTS    
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    '''
    def setup(self,bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.thresh = 5 # threshold on ab value
        self.N = bottom[0].data.shape[0]
        self.X = bottom[0].data.shape[2]
        self.Y = bottom[0].data.shape[3]

    def reshape(self, bottom, top):
        top[0].reshape(self.N,1,self.X,self.Y)
 
    def forward(self, bottom, top):
        # if an image has any (a,b) value which exceeds threshold, output 1
        top[0].data[...] = (np.sum(np.sum(np.sum(np.abs(bottom[0].data) > self.thresh,axis=1),axis=1),axis=1) > 0)[:,na(),na(),na()]

    def backward(self, top, propagate_down, bottom):
        # no back-prop
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[i].diff[...] = np.zeros_like(bottom[i].data)

class ClassRebalanceMultLayer(caffe.Layer):
# '''
# INPUTS
#     bottom[0]   NxMxXxY     feature map
#     bottom[1]   Nx1xXxY     boost coefficients
# OUTPUTS
#     top[0]      NxMxXxY     on forward, gets copied from bottom[0]
# FUNCTIONALITY
#     On forward pass, top[0] passes bottom[0]
#     On backward pass, bottom[0] gets boosted by bottom[1]
#     through pointwise multiplication (with singleton expansion) '''
    def setup(self, bottom, top):
        # check input pair
        if len(bottom)==0:
            raise Exception("Specify inputs")

    def reshape(self, bottom, top):
        i = 0
        if(bottom[i].data.ndim==1):
            top[i].reshape(bottom[i].data.shape[0])
        elif(bottom[i].data.ndim==2):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1])
        elif(bottom[i].data.ndim==4):
            top[i].reshape(bottom[i].data.shape[0], bottom[i].data.shape[1], bottom[i].data.shape[2], bottom[i].data.shape[3])

    def forward(self, bottom, top):
        # output equation to negative of inputs
        top[0].data[...] = bottom[0].data[...]
        # top[0].data[...] = bottom[0].data[...]*bottom[1].data[...] # this was bad, would mess up the gradients going up

    def backward(self, top, propagate_down, bottom):
        for i in range(len(bottom)):
            if not propagate_down[i]:
                continue
            bottom[0].diff[...] = top[0].diff[...]*bottom[1].data[...]
            # print 'Back-propagating class rebalance, %i'%i

# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor '''
    # def __init__(self,alpha,gamma=0,verbose=True,priorFile='/home/eecs/rich.zhang/src/projects/cross_domain/save/ab_grid_10/prior_probs.npy',genc=-1):
    def __init__(self,alpha,gamma=0,verbose=True,priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior^alpha power
        #   gamma           integer     percentage to mix in prior probability
        #   priorFile       file        file which contains prior probabilities across classes    

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs*self.prior_factor
        self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        # add this to the softmax score
        # self.softmax_correction = np.log(self.prior_probs/self.implied_prior * (1-self.implied_prior)/(1-self.prior_probs))

        if(self.verbose):
            self.print_correction_stats()

        # if(not check_value(genc,-1)):
            # self.expand_grid(genc)

    # def expand_grid(self,genc):
    #     self.prior_probs_full_grid = genc.enc_full_grid_mtx_nd(self.prior_probs,axis=0,returnGrid=True)
    #     self.uni_probs_full_grid = genc.enc_full_grid_mtx_nd(self.uni_probs,axis=0,returnGrid=True)
    #     self.prior_mix_full_grid = genc.enc_full_grid_mtx_nd(self.prior_mix,axis=0,returnGrid=True)
    #     self.prior_factor_full_grid = genc.enc_full_grid_mtx_nd(self.prior_factor,axis=0,returnGrid=True)
    #     self.implied_prior_full_grid = genc.enc_full_grid_mtx_nd(self.implied_prior,axis=0,returnGrid=True)
    #     self.softmax_correction_full_grid = genc.enc_full_grid_mtx_nd(self.softmax_correction,axis=0,returnGrid=True)

    def print_correction_stats(self):
        print 'Prior factor correction:'
        print '  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma)
        print '  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)'%(np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs))

    def forward(self,data_ab_quant,axis=1):
        # data_ab_quant = net.blobs['data_ab_quant_map_233'].data[...]
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:]
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]

class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        # self.NN = NN
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

    def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False):
        pts_flt = flatten_nd_array(pts_nd,axis=axis)
        P = pts_flt.shape[0]

        (dists,inds) = self.nbrs.kneighbors(pts_flt)

        pts_enc_flt = np.zeros((P,self.K))
        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

        pts_enc_flt[np.arange(0,P,dtype='int')[:,na()],inds] = wts
        pts_enc_nd = unflatten_2d_array(pts_enc_flt,pts_nd,axis=axis)

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd


# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd         N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd         N0xN1x...xNd array
        axis         integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        # print NEW_SHP
        # print pts_flt.shape
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out