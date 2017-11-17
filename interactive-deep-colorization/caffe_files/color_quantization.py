 
import numpy as np
from IPython.core.debugger import Pdb as pdb
import sklearn.neighbors as nn
import util
import caffe

class NNEncode():
	# Encode points as a linear combination of unordered points
	# using NN search and RBF kernel
	def __init__(self,NN,sigma,km_filepath='./data/color_bins/pts_in_hull.npy',cc=-1):
		if(util.check_value(cc,-1)):
			self.cc = np.load(km_filepath)
		else:
			self.cc = cc
		self.K = self.cc.shape[0]
		self.NN = int(NN)
		self.sigma = sigma
		self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='auto').fit(self.cc)

	def encode_points_mtx_nd(self,pts_nd,axis=1,returnSparse=False):
		t = util.Timer()
		pts_flt = util.flatten_nd_array(pts_nd,axis=axis)
		P = pts_flt.shape[0]

		(dists,inds) = self.nbrs.kneighbors(pts_flt)

		pts_enc_flt = np.zeros((P,self.K))
		wts = np.exp(-dists**2/(2*self.sigma**2))
		wts = wts/np.sum(wts,axis=1)[:,util.na()]

		pts_enc_flt[np.arange(0,P,dtype='int')[:,util.na()],inds] = wts
		pts_enc_nd = util.unflatten_2d_array(pts_enc_flt,pts_nd,axis=axis)

		return pts_enc_nd

	def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
		pts_enc_flt = util.flatten_nd_array(pts_enc_nd,axis=axis)
		pts_dec_flt = np.dot(pts_enc_flt,self.cc)
		pts_dec_nd = util.unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
		return pts_dec_nd

	def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
		pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
		pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
		if(returnEncode):
			return (pts_dec_nd,pts_1hot_nd)
		else:
			return pts_dec_nd

