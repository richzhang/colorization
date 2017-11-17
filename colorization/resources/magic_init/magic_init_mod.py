from __future__ import print_function, division

INPUT_LAYERS = ['Data', 'ImageData']
# Layers that only support elwise
ELWISE_LAYERS = ['Deconvolution']
# Layers that support parameters
PARAMETER_LAYERS = ['Convolution', 'InnerProduct']+ELWISE_LAYERS
# All supported layers
SUPPORTED_LAYERS = ['ReLU', 'Sigmoid', 'LRN', 'Pooling', 'Eltwise'] + PARAMETER_LAYERS + INPUT_LAYERS
STRIP_LAYER = ['Softmax', 'SoftmaxWithLoss', 'SigmoidCrossEntropyLoss']
# Use 'Dropout' at your own risk
# Unless Jon merges #2865 , 'Split' cannot be supported
UNSUPPORTED_LAYERS = ['Split', 'BatchNorm', 'Reshape', 'Scale']

def forward(net, i, NIT, data, output_names):
	n = net._layer_names[i]
	# Create the top data if needed
	output = {t: [None]*NIT for t in output_names}
	for it in range(NIT):
		for b in data:
			net.blobs[b].data[...] = data[b][it]
		net._forward(i, i)
		for t in output_names:
			output[t][it] = 1*net.blobs[t].data
	return output

def flattenData(data):
	import numpy as np
	return np.concatenate([d.swapaxes(0, 1).reshape((d.shape[1],-1)) for d in data], axis=1).T

def gatherInputData(net, layer_id, bottom_data, top_name, fast=False, max_data=None):
	# This functions gathers all input data.
	# In order to not replicate all the internal functionality of convolutions (eg. padding ...)
	# we gather the data in the output space and use random gaussian weights. The output of this
	# function is W and D, there the input data I = D * W^-1  [with some abuse of tensor notation]
	# If we not compute an initialization A for D, we then simply multiply A by W to obtain the
	# proper initialization in the input space
	import numpy as np
	l = net.layers[layer_id]
	NIT = len(list(bottom_data.values())[0])
	
	# How many times do we need to over-sample to get a full basis (out of random projections)
	OS = int(np.ceil( np.prod(l.blobs[0].data.shape[1:]) / l.blobs[0].data.shape[0] ))
	if fast: OS = 1
	
	# If we are over sampling we might run out of memory at some point, especially for filters higher up
	# Do avoid any issues we never return more than max_data number of elements
	subsample = None
	
	# Note this could cause some memory issues in the FC layers
	W, D = [], []
	for i in range(OS):
		d = l.blobs[0].data
		d[...] = np.random.normal(0, 1, d.shape)
		W.append(1*d)
		# Collect the data and flatten out the convs
		data = np.concatenate([i.swapaxes(0, 1).reshape((i.shape[1],-1)).T for i in forward(net, layer_id, NIT, bottom_data, [top_name])[top_name]], axis=0)
		# Do we need to subsample the data to save memory?
		if subsample is None and max_data is not None:
			# Randomly select n data representative samples
			N = int(max_data / (data.shape[1]*OS))
			subsample = np.arange(data.shape[0])
			if N < data.shape[0]:
				np.random.shuffle(subsample)
				subsample = subsample[:N]
		if subsample is not None:
			data = data[subsample]
		D.append(data)
	# In order to handle any sort of groups we want to have the samples packed in the following order:
	# a1 a2 a3 a4 b1 b2 b3 b4 c1 ...  (where the original data was a b c and OS=4)
	W, D = np.concatenate([w[:,None] for w in W], axis=1), np.concatenate([d[:,:,None] for d in D], axis=2)
	return W.reshape((-1,)+W.shape[2:]), D.reshape((D.shape[0], -1)+D.shape[3:])

def initializeWeight(D, type, N_OUT):
	# Here we first whiten the data (PCA or ZCA) and then optionally run k-means
	# on this whitened data.
	import numpy as np
	if D.shape[0] < N_OUT:
		print( "  Not enough data for '%s' estimation, using elwise"%type )
		return np.random.normal(0, 1, (N_OUT,D.shape[1]))
	D = D - np.mean(D, axis=0, keepdims=True)
	# PCA, ZCA, K-Means
	assert type in ['pca', 'zca', 'kmeans', 'rand'], "Unknown initialization type '%s'"%type
	C = D.T.dot(D)
	s, V = np.linalg.eigh(C)
	# order the eigenvalues
	ids = np.argsort(s)[-N_OUT:]
	s = s[ids]
	V = V[:,ids]
	s[s<1e-6] = 0
	s[s>=1e-6] = 1. / np.sqrt(s[s>=1e-6]+1e-3)
	S = np.diag(s)
	if type == 'pca':
		return S.dot(V.T)
	elif type == 'zca':
		return V.dot(S.dot(V.T))
	# Whiten the data
	wD = D.dot(V.dot(S))
	wD /= np.linalg.norm(wD, axis=1)[:,None]
	if type == 'kmeans':
		# Run k-means
		from sklearn.cluster import MiniBatchKMeans
		km = MiniBatchKMeans(n_clusters = wD.shape[1], batch_size=10*wD.shape[1]).fit(wD).cluster_centers_
	elif type == 'rand':
		km = wD[np.random.choice(wD.shape[0], wD.shape[1], False)]
	C = km.dot(S.dot(V.T))
	C /= np.std(D.dot(C.T), axis=0, keepdims=True).T
	return C
		

def initializeLayer(net, layer_id, bottom_data, top_name, bias=0, type='elwise', max_data=None):
	import numpy as np
	l = net.layers[layer_id]
	NIT = len(list(bottom_data.values())[0])
	
	if type!='elwise' and l.type in ELWISE_LAYERS:
		print( "Only 'elwise' supported for layer '%s'. Falling back."%net._layer_names[layer_id] )
		type = 'elwise'
	
	for p in l.blobs: p.data[...] = 0
	fast = 'fast_' in type
	if fast:
		type = type.replace('fast_', '')
	# Initialize the weights [k-means, ...]
	if type == 'elwise':
		d = l.blobs[0].data
		d[...] = np.random.normal(0, 1, d.shape)
	else: # Use the input data
		# Are there any groups?
		G = 1
		bottom_names = net.bottom_names[net._layer_names[layer_id]]
		if len(bottom_names) == 1:
			N1 = net.blobs[bottom_names[0]].shape[1]
			N2 = l.blobs[0].shape[1]
			G = N1 // N2
		
		# Gather the input data
		print( "  Gathering input data")
		T, D = gatherInputData(net, layer_id, bottom_data, top_name, fast, max_data=max_data)

		# Figure out the output dimensionality of d
		d = l.blobs[0].data

		print( "  Initializing weights" )
		# Loop over groups
		for g in range(G):
			dg, Dg = d[g*(d.shape[0]//G):(g+1)*(d.shape[0]//G)], D[:,g*(D.shape[1]//G):(g+1)*(D.shape[1]//G):]
			Tg = T[g*(T.shape[0]//G):(g+1)*(T.shape[0]//G)]
			# Compute the weights
			W = initializeWeight(Dg, type, N_OUT=dg.shape[0])
			
			# Multiply the weights by the random basis
			# NOTE: This matrix multiplication is a bit large, if it's too slow,
			#       reduce the oversampling in gatherInputData
			dg[...] = np.dot(W, Tg.reshape((Tg.shape[0],-1))).reshape(dg.shape)

	# Scale the mean and initialize the bias
	print( "  Scale the mean and initialize the bias" )
	top_data = forward(net, layer_id, NIT, bottom_data, [top_name])[top_name]
	flat_data = flattenData(top_data)
	mu = flat_data.mean(axis=0)
	std = flat_data.std(axis=0)
	for ii in range(np.minimum(mu.size,5)):
		print("  mu+/-std: (%.3f,%.3f)"%(mu[ii],std[ii]))
	if l.type == 'Deconvolution':
		l.blobs[0].data[...] /= std.reshape((1,-1,)+(1,)*(len(l.blobs[0].data.shape)-2))
	else:
		l.blobs[0].data[...] /= std.reshape((-1,)+(1,)*(len(l.blobs[0].data.shape)-1))
	for b in l.blobs[1:]:
		b.data[...] = -mu / std + bias



def magicInitialize(net, bias=0, NIT=10, type='elwise', max_data=None):
	import numpy as np
	# When was a blob last used
	last_used = {}
	# Make sure all layers are supported, and compute the last time each blob is used
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		if l.type in UNSUPPORTED_LAYERS:
			print( "WARNING: Layer type '%s' not supported! Things might go very wrong..."%l.type )
		elif l.type not in SUPPORTED_LAYERS+STRIP_LAYER:
			print( "Unknown layer type '%s'. double check if it is supported"%l.type )
		for b in net.bottom_names[n]:
			last_used[b] = i
	
	active_data = {}
	# Read all the input data
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		# Initialize the layer
		if (len(l.blobs) > 0) and (l.type not in UNSUPPORTED_LAYERS):
		# if len(l.blobs) > 0:
			if np.sum(np.abs(l.blobs[0].data)) <= 1e-10:
				print( "Initializing layer '%s'"%n )
				assert l.type in PARAMETER_LAYERS, "Unsupported parameter layer"
				assert len(net.top_names[n]) == 1, "Exactly one output supported"
				
				# Fill the parameters
				initializeLayer(net, i, {b: active_data[b] for b in net.bottom_names[n]}, net.top_names[n][0], bias, type, max_data=max_data)
			else:
				print( "Skipping layer '%s'"%n )
		else:
			print( "Skipping layer '%s'"%n )
			# TODO: Estimate and rescale the values [TODO: Record and undo this scaling above]
		
		# Run the network forward
		new_data = forward(net, i, NIT, {b: active_data[b] for b in net.bottom_names[n]}, net.top_names[n])
		active_data.update(new_data)
		
		# Delete all unused data
		for k in list(active_data):
			if k not in last_used or last_used[k] == i:
				del active_data[k]

def load(net, blobs):
	for l,n in zip(net.layers, net._layer_names):
		if n in blobs:
			for b, sb in zip(l.blobs, blobs[n]):
				b.data[...] = sb

def save(net):
	import numpy as np
	r = {}
	for l,n in zip(net.layers, net._layer_names):
		if len(l.blobs) > 0:
			r[n] = [np.copy(b.data) for b in l.blobs]
	return r

def estimateHomogenety(net):
	# Estimate if a certain layer is homogeneous and if yes return the degree k
	# by which the output is scaled (if input is scaled by alpha then the output
	# is scaled by alpha^k). Return None if the layer is not homogeneous.
	import numpy as np

	print("Estimating homogenety")

	# When was a blob last used
	last_used = {}
	# Make sure all layers are supported, and compute the range each blob is used in
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		for b in net.bottom_names[n]:
			last_used[b] = i
	
	active_data = {}
	homogenety = {}
	# Read all the input data
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		# Run the network forward
		new_data1 = forward(net, i, 1, {b: [1*d for d in active_data[b]] for b in net.bottom_names[n]}, net.top_names[n])
		new_data2 = forward(net, i, 1, {b: [2*d for d in active_data[b]] for b in net.bottom_names[n]}, net.top_names[n])
		active_data.update(new_data1)
		
		if len(new_data1) == 1:
			m = list(new_data1.keys())[0]
			d1, d2 = flattenData(new_data1[m]), flattenData(new_data2[m])
			f = np.mean(np.abs(d1), axis=0) / np.mean(np.abs(d2), axis=0)
			if 1e-3*np.mean(f) < np.std(f):
				# Not homogeneous
				homogenety[n] = None
			else:
				# Compute the degree of the homogeneous transformation
				homogenety[n] = (np.log(np.mean(np.abs(d2))) - np.log(np.mean(np.abs(d1)))) / np.log(2)
		else:
			homogenety[n] = None
		# Delete all unused data
		for k in list(active_data):
			if k not in last_used or last_used[k] == i:
				del active_data[k]
	return homogenety

def calibrateGradientRatio(net, NIT=1):
	print('Calibrate gradient ratio')

	import numpy as np
	# When was a blob last used
	last_used = {}
	# Find the last layer to use
	last_layer = 0
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		print('\tLayer %s'%n)
		if l.type not in STRIP_LAYER:
			last_layer = i
		for b in net.bottom_names[n]:
			last_used[b] = i
	# Figure out which tops are involved
	last_tops = net.top_names[net._layer_names[last_layer]]
	for t in last_tops:
		last_used[t] = len(net.layers)
	
	# Call forward and store the data of all data layers
	print('Call forward and store the data of all data layers')
	active_data, input_data, bottom_scale = {}, {}, {}
	# Read all the input data
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		print('\tLayer %s'%n)

		if i > last_layer: break
		# Compute the input scale for parameter layers
		if len(l.blobs) > 0:
			bottom_scale[n] = np.mean([np.mean(np.abs(active_data[b])) for b in net.bottom_names[n]])
		# Run the network forward
		new_data = forward(net, i, NIT, {b: active_data[b] for b in net.bottom_names[n]}, net.top_names[n])
		if l.type in INPUT_LAYERS:
			input_data.update(new_data)
		active_data.update(new_data)
		
		# Delete all unused data
		for k in list(active_data):
			if k not in last_used or last_used[k] == i:
				del active_data[k]

	output_std = np.mean(np.std(flattenData(active_data[last_tops[0]]), axis=0))
	
	for it in range(10):
	# for it in range(1):
		print('Iteration %i'%it)
		# Reset the diffs
		for l in net.layers:
			for b in l.blobs:
				b.diff[...] = 0
		# Set the top diffs
		print('Last layer')
		print(last_tops)
		print(last_layer)

		for t in last_tops:
			print(t)
			net.blobs[t].diff[...] = np.random.normal(0, 1, net.blobs[t].shape)

		# Compute all gradients
		# print(np.mean(net.blobs[t].diff[...]**2))
		# print(np.mean(net.blobs[t].data[...]**2))
		net._backward(last_layer, 0)
		# # net.backward()
		# print(np.mean(net.blobs[t].diff[...]**2))
		# print(np.mean(net.blobs[t].data[...]**2))

		# print(np.mean(net.blobs['da_conv1'].data[...]**2))
		
		# Compute the gradient ratio
		ratio={}
		for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
			print('layer index %i, layer name %s'%(i,n))
			if len(l.blobs) > 0:
			# if (len(l.blobs) > 0) and (l.type in PARAMETER_LAYERS):
				assert l.type in PARAMETER_LAYERS, "Parameter layer '%s' currently not supported"%l.type
				b = l.blobs[0]
				ratio[n] = np.sqrt(np.mean(b.diff**2) / np.mean(b.data**2))
				print('Ratio = sqrt(diff/data), %.0f=sqrt(%.3e/%3e)'%(ratio[n],np.mean(b.diff**2),np.mean(b.data**2)))

		# print(ratio)

		# If all layers are homogeneous, then the target ratio is the geometric mean of all ratios
		# (assuming we want the same output)
		# To deal with non-homogeneous layers we scale by output_std in the hope to undo correct the
		# estimation over time.
		# NOTE: for non feed-forward networks the geometric mean might not be the right scaling factor

		target_ratio = np.exp(np.mean(np.log(np.array(list(ratio.values()))))) * (output_std)**(1. / len(ratio))
		for val in np.array(list(ratio.values())):
			print(val)

		# np.exp(np.mean(np.log(np.array(list(ratio.values())))))
		# (output_std)**(1. / len(ratio))

		# print(len(ratio))
		print('Num ratios: %i'%len(ratio))
		print('Target ratio: %.0f'%target_ratio)
		print('Current ratios (mean/std): %.0f+/-%.0f'%(np.mean(np.array(list(ratio.values()))),np.std(np.array(list(ratio.values())))))
		
		# Terminate if the relative change is less than 1% for all values
		log_ratio = np.log( np.array(list(ratio.values())) )
		print('Max relative change: %.3f'%np.max(np.abs(log_ratio/np.log(target_ratio)-1)))
		if np.all( np.abs(log_ratio/np.log(target_ratio) - 1) < 0.01 ):
			break
		
		# Update all the weights and biases
		active_data = {}
		# Read all the input data
		for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
			if i > last_layer: break
			# Use the stored input
			if l.type in INPUT_LAYERS:
				active_data.update({b: input_data[b] for b in net.top_names[n]})
			else:
				if len(l.blobs) > 0:
				# if (len(l.blobs) > 0) and (l.type in PARAMETER_LAYERS):
					# Add the scaling from the bottom to the biases
					current_scale = np.mean([np.mean(np.abs(active_data[b])) for b in net.bottom_names[n]])
					adj = current_scale / bottom_scale[n]
					for b in list(l.blobs)[1:]:
						b.data[...] *= adj
					bottom_scale[n] = current_scale
					
					# Scale to obtain the target ratio
					scale = np.sqrt(ratio[n] / target_ratio)
					for b in l.blobs:
						b.data[...] *= scale
					
				active_data.update(forward(net, i, NIT, {b: active_data[b] for b in net.bottom_names[n]}, net.top_names[n]))
			# Delete all unused data
			for k in list(active_data):
				if k not in last_used or last_used[k] == i:
					del active_data[k]

		new_output_std = np.mean(np.std(flattenData(active_data[last_tops[0]]), axis=0))
		print(np.linalg.norm(active_data[last_tops[0]]))
		print(last_tops[0])
		print(new_output_std)
		if np.abs(np.log(output_std) - np.log(new_output_std)) > 0.25:
			# If we diverge by a factor of exp(0.25) = ~1.3, then we should check if the network is really
			# homogeneous
			print( "WARNING: It looks like one or more layers are not homogeneous! Trying to correct for this..." )
			print( "         Output std = %f" % new_output_std )
		output_std = new_output_std

	print('')

def netFromString(s, t=None):
	import caffe
	from tempfile import NamedTemporaryFile
	if t is None: t = caffe.TEST
	f = NamedTemporaryFile('w')
	f.write(s)
	f.flush()
	r = caffe.Net(f.name, t)
	f.close()
	return r

def getFileList(f):
	from glob import glob
	from os import path
	return [f for f in glob(f) if path.isfile(f)]

def main():
	from argparse import ArgumentParser
	from os import path
	import numpy as np
	parser = ArgumentParser()
	parser.add_argument('prototxt')
	parser.add_argument('output_caffemodel')
	parser.add_argument('-l', '--load', help='Load a pretrained model and rescale it [bias and type are not supported]')
	parser.add_argument('-d', '--data', default=None, help='Image list to use [default prototxt data]')
	parser.add_argument('-b', '--bias', type=float, default=0.1, help='Bias')
	parser.add_argument('-t', '--type', default='elwise', help='Type: elwise, pca, zca, kmeans, rand (random input patches). Add fast_ to speed up the initialization, but you might lose in precision.')
	parser.add_argument('-z', action='store_true', help='Zero all weights and reinitialize')
	parser.add_argument('-cs',  action='store_true', help='Correct for scaling')
	parser.add_argument('-q', action='store_true', help='Quiet execution')
	parser.add_argument('-s', type=float, default=1.0, help='Scale the input [only custom data "-d"]')
	parser.add_argument('-bs', type=int, default=16, help='Batch size [only custom data "-d"]')
	parser.add_argument('-nit', type=int, default=10, help='Number of iterations')
	parser.add_argument('--mem-limit', type=int, default=500, help='How much memory should we use for the data buffer (MB)?')
	parser.add_argument('--gpu', type=int, default=0, help='What gpu to run it on?')
	args = parser.parse_args()
	
	if args.q:
		from os import environ
		environ['GLOG_minloglevel'] = '2'
	import caffe, load
	from caffe import NetSpec, layers as L
	
	caffe.set_mode_gpu()
	if args.gpu is not None:
		caffe.set_device(args.gpu)
	
	if args.data is not None:
		model = load.ProtoDesc(args.prototxt)
		net = NetSpec()
		fl = getFileList(args.data)
		if len(fl) == 0:
			print("Unknown data type for '%s'"%args.data)
			exit(1)
		from tempfile import NamedTemporaryFile
		f = NamedTemporaryFile('w')
		f.write('\n'.join([path.abspath(i)+' 0' for i in fl]))
		f.flush()
		net.data, net.label = L.ImageData(source=f.name, batch_size=args.bs, new_width=model.input_dim[-1], new_height=model.input_dim[-1], transform_param=dict(mean_value=[104,117,123], scale=args.s),ntop=2)
		net.out = model(data=net.data, label=net.label)
		n = netFromString('force_backward:true\n'+str(net.to_proto()), caffe.TRAIN )
	else:
		n = caffe.Net(args.prototxt, caffe.TRAIN)

	# forward call on network
	n.forward()

	if args.load is not None:
		n.copy_from(args.load)
		# Rescale existing layers?
		#if args.fix:
			#magicFix(n, args.nit)

	if args.z:
		# Zero out all layers
		for l in n.layers:
			for b in l.blobs:
				b.data[...] = 0
	if any([np.abs(l.blobs[0].data).sum() < 1e-10 for l in n.layers if len(l.blobs) > 0]):
		print( [m for l,m in zip(n.layers, n._layer_names) if len(l.blobs) > 0 and np.abs(l.blobs[0].data).sum() < 1e-10] )
		magicInitialize(n, args.bias, NIT=args.nit, type=args.type, max_data=args.mem_limit*1024*1024/4)
	else:
		print( "Network already initialized, skipping magic init" )
	if args.cs:
		# A simply helper function that lets you figure out which layers are not
		# homogeneous
		# print( estimateHomogenety(n) )

		calibrateGradientRatio(n)
	n.save(args.output_caffemodel)

if __name__ == "__main__":
	main()
