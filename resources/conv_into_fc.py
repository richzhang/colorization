
import caffe
import os
import string
import numpy as np
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Convert conv layers into FC layers')
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--prototxt_in',dest='prototxt_in',help='prototxt with conv layers', type=str, default='')
    parser.add_argument('--prototxt_out',dest='prototxt_out',help='prototxt with fc layers', type=str, default='')
    parser.add_argument('--caffemodel_in',dest='caffemodel_in',help='caffemodel with conv layers', type=str, default='')
    parser.add_argument('--caffemodel_out',dest='caffemodel_out',help='caffemodel with fc layers, to be saved', type=str, default='')
    parser.add_argument('--dummymodel',dest='dummymodel',help='blank caffemodel',type=str,default='./models/dummy.caffemodel')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = parse_args()

	gpu_id = args.gpu
	PROTOTXT1_PATH = args.prototxt_in
	PROTOTXT2_PATH = args.prototxt_out # no batch norm
	MODEL_PATH = args.caffemodel_in	
	DUMMYMODEL_PATH = args.dummymodel
	MODEL2_PATH = args.caffemodel_out # to be saved off

	caffe.set_mode_gpu()
	caffe.set_device(gpu_id)

	net1 = caffe.Net(PROTOTXT1_PATH, MODEL_PATH, caffe.TEST)
	net2 = caffe.Net(PROTOTXT2_PATH, DUMMYMODEL_PATH, caffe.TEST)

	import rz_fcns as rz
	rz.caffe_param_shapes(net1,to_print=True)
	rz.caffe_param_shapes(net2,to_print=True)
	rz.caffe_shapes(net2,to_print=True)

	# CONV_INDS = np.where(np.array([layer.type for layer in net1.layers])=='Convolution')[0]
	print net1.params.keys()
	print net2.params.keys()

	for (ll,layer) in enumerate(net2.params.keys()):
		P = len(net2.params[layer]) # number of blobs
		if(P>0):
			for pp in range(P):
				ndim1 = net1.params[layer][pp].data.ndim
				ndim2 = net2.params[layer][pp].data.ndim

				print('Copying layer %s, param blob %i (%i-dim => %i-dim)'%(layer,pp,ndim1,ndim2))
				if(ndim1==ndim2):
					print('  Same dimensionality...')
					net2.params[layer][pp].data[...] = net1.params[layer][pp].data[...]
				else:
					print('  Different dimensionality...')
					net2.params[layer][pp].data[...] = net1.params[layer][pp].data[...].reshape(net2.params[layer][pp].data[...].shape)

	net2.save(MODEL2_PATH)

	for arg in vars(args):
		print('[%s] =' % arg, getattr(args, arg))
	print 'Saving model into: %s'%MODEL2_PATH
