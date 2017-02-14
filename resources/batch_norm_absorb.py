
# **************************************
# ***** Richard Zhang / 2016.06.04 *****
# **************************************
# Absorb batch norm into convolution layers
# This script only supports the conv-batchnorm configuration
# Currently unsupported: 
# 	- deconv layers
# 	- fc layers
# 	- batchnorm before linear layer

import caffe
import os
import string
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='BatchNorm absorption')
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--prototxt_in',dest='prototxt_in',help='prototxt with batchnorm', type=str, default='')
    parser.add_argument('--prototxt_out',dest='prototxt_out',help='prototxt without batchnorm', type=str, default='')
    parser.add_argument('--caffemodel_in',dest='caffemodel_in',help='caffemodel with batchnorm', type=str, default='')
    parser.add_argument('--caffemodel_out',dest='caffemodel_out',help='caffemodel without batchnorm, to be saved', type=str, default='')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_args()

	gpu_id = args.gpu
	PROTOTXT1_PATH = args.prototxt_in
	PROTOTXT2_PATH = args.prototxt_out # no batch norm
	MODEL_PATH = args.caffemodel_in
	MODEL2_PATH = args.caffemodel_out # to be saved off

	caffe.set_mode_gpu()
	caffe.set_device(gpu_id)

	net1 = caffe.Net(PROTOTXT1_PATH, MODEL_PATH, caffe.TEST)
	net2 = caffe.Net(PROTOTXT2_PATH, MODEL_PATH, caffe.TEST)

	# call forward on net1, net2
	net1.forward()
	net2.forward()

	# identify batch norms and paired linear layers
	BN_INDS = np.where(np.array([layer.type for layer in net1.layers])=='BatchNorm')[0]
	BN_NAMES = np.zeros(BN_INDS.shape,dtype='S50') # batch norm layer names
	LIN_NAMES = np.zeros(BN_INDS.shape,dtype='S50') # linear layer names
	PRE_NAMES = np.zeros(BN_INDS.shape,dtype='S50') # blob right before
	POST_NAMES = np.zeros(BN_INDS.shape,dtype='S50') # blob right after

	PRE_POST = -1+np.zeros(BN_INDS.shape) # 0 - pre, 1 - post
	CONV_DECONV = -1+np.zeros(BN_INDS.shape) # 0 - conv, 1 - deconv

	# identify layers which are paired with batch norms (only supporting convolution)
	for (ll,bn_ind) in enumerate(BN_INDS):
		BN_NAMES[ll] = net1._layer_names[bn_ind]
		if(net1.layers[bn_ind-1].type=='Convolution' or net1.layers[bn_ind-1].type=='Deconvolution'):
			PRE_POST[ll] = 0
			LIN_NAMES[ll] = net1._layer_names[bn_ind-1]
			POST_NAMES[ll] = net1._layer_names[bn_ind+1]
			if(net1.layers[bn_ind-1].type=='Convolution'):
				CONV_DECONV[ll] = 0
			elif(net1.layers[bn_ind-1].type=='Deconvolution'):
				CONV_DECONV[ll] = 1
		elif(net1.layers[bn_ind+1].type=='Convolution' or net1.layers[bn_ind+1].type=='Deconvolution'):
			PRE_POST[ll] = 1 
			LIN_NAMES[ll] = net1._layer_names[bn_ind+1]
			POST_NAMES[ll] = net1._layer_names[bn_ind+3]
			if(net1.layers[bn_ind+1].type=='Convolution'):
				CONV_DECONV[ll] = 0
			elif(net1.layers[bn_ind+1].type=='Deconvolution'):
				CONV_DECONV[ll] = 1
		else:
			PRE_POST[ll] = -1
		PRE_NAMES[ll] = net1.bottom_names[BN_NAMES[ll]][0]

	LIN_INDS = BN_INDS+PRE_POST # linear layer indices
	ALL_SLOPES = {}

	# compute batch norm parameters on net1 in first layer
	# absorb into weights in first layer
	for ll in range(BN_INDS.size):
		bn_ind = BN_INDS[ll]
		BN_NAME = BN_NAMES[ll]
		PRE_NAME = PRE_NAMES[ll]
		POST_NAME = POST_NAMES[ll]
		LIN_NAME = LIN_NAMES[ll]

		print 'LAYERS %s, %s'%(PRE_NAME,BN_NAME)
		# print net1.blobs[BN_NAME].data.shape
		# print net1.blobs[PRE_NAME].data.shape

		C = net1.blobs[BN_NAME].data.shape[1]
		in_blob = net1.blobs[PRE_NAME].data
		bn_blob = net1.blobs[BN_NAME].data

		scale_factor = 1./net1.params[BN_NAME][2].data[...]
		mean = scale_factor * net1.params[BN_NAME][0].data[...]
		scale = scale_factor * net1.params[BN_NAME][1].data[...]

		slopes = np.sqrt(1./scale)
		offs = -mean*slopes

		print '  Computing error on data...'
		bn_blob_rep = in_blob*slopes[np.newaxis,:,np.newaxis,np.newaxis]+offs[np.newaxis,:,np.newaxis,np.newaxis]

		# Visually verify that factors are correct
		print '  Maximum error: %.3e'%np.max(np.abs(bn_blob_rep[bn_blob>0] - bn_blob[bn_blob>0]))
		print '  RMS error: %.3e'%np.linalg.norm(bn_blob_rep[bn_blob>0] - bn_blob[bn_blob>0])
		print '  RMS signal: %.3e'%np.linalg.norm(bn_blob_rep[bn_blob>0])

		print '  Absorbing slope and offset...'
		# absorb slope and offset into appropriate parameter
		if(PRE_POST[ll]==0): # linear layer is before
			if(CONV_DECONV[ll]==0): # convolution
				net2.params[LIN_NAME][0].data[...] = net1.params[LIN_NAME][0].data[...]*slopes[:,np.newaxis,np.newaxis,np.newaxis]
				net2.params[LIN_NAME][1].data[...] = offs + (slopes*net1.params[LIN_NAME][1].data)
			elif(CONV_DECONV[ll]==1): # deconvolution
				print '*** Deconvolution not implemented ***'
		elif(PRE_POST[ll]==1): # batchnorm is BEFORE linear layer
			print '*** Not implemented ***'

	net2.save(MODEL2_PATH)

	for arg in vars(args):
		print('[%s] =' % arg, getattr(args, arg))
	print 'Saving model into: %s'%MODEL2_PATH
