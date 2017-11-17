from __future__ import print_function
from magic_init import *

class BCOLORS:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	
class NOCOLORS:
	HEADER = ''
	OKBLUE = ''
	OKGREEN = ''
	WARNING = ''
	FAIL = ''
	ENDC = ''
	BOLD = ''
	UNDERLINE = ''

def coloredNumbers(v, color=None, fmt='%6.2f', max_display=300, bcolors=BCOLORS):
	import numpy as np
	# Display a numpy array and highlight the min and max values [required a nice linux
	# terminal supporting colors]
	r = ""
	mn, mx = np.min(v), np.max(v)
	for k,i in enumerate(v):
		if len(v) > max_display and k > max_display/2 and k < len(v) - max_display/2:
			if r[-1] != '.':
				r += '...'
			continue
		if i <= mn + 1e-3:
			r += bcolors.BOLD+bcolors.FAIL
		elif i + 1e-3 >= mx:
			r += bcolors.BOLD+bcolors.FAIL
		elif color is not None:
			r += color
		r += (fmt+' ')%i
		r += bcolors.ENDC
	r += bcolors.ENDC
	return r

def computeGradientRatio(net, NIT=1):
	import numpy as np
	last_layer = 0
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		if l.type not in STRIP_LAYER:
			last_layer = i
	last_tops = net.top_names[net._layer_names[last_layer]]
	
	var = {}
	for it in range(NIT):
		net._forward(0, last_layer)
		# Reset the diffs
		for l in net.layers:
			for b in l.blobs:
				b.diff[...] = 0
		# Set the top diffs
		for t in last_tops:
			net.blobs[t].diff[...] = np.random.normal(0, 1, net.blobs[t].shape)
		net._backward(last_layer, 0)
		for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
			if len(l.blobs) > 0:
				assert l.type in PARAMETER_LAYERS, "Parameter layer '%s' currently not supported"%l.type
				b = l.blobs[0]
				r = np.mean(b.diff.swapaxes(0,1).reshape((b.diff.shape[1],-1))**2, axis=1) / np.mean(b.data**2)
				if n in var: var[n] += r / NIT
				else: var[n] = r / NIT
	std = {n: np.sqrt(var[n]) for n in var}
	return {n: np.std(s) / np.mean(s) for n,s in std.items()}, {n: np.mean(s) for n,s in std.items()}

def printMeanStddev(net, NIT=10, show_all=False, show_color=True, quiet=False):
	import numpy as np
	bcolors = NOCOLORS
	if show_color: bcolors = BCOLORS
	
	layer_names = list(net._layer_names)
	if not show_all:
		layer_names = [n for n, l in zip(net._layer_names, net.layers) if len(l.blobs)>0]
		if 'data' in net._layer_names:
			layer_names.append('data')

	# When was a blob last used
	last_used = {}
	# Make sure all layers are supported, and compute the range each blob is used in
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		for b in net.bottom_names[n]:
			last_used[b] = i
	
	active_data, cvar = {}, {}
	for i, (n, l) in enumerate(zip(net._layer_names, net.layers)):
		# Run the network forward
		new_data = forward(net, i, NIT, {b: active_data[b] for b in net.bottom_names[n]}, net.top_names[n])
		active_data.update(new_data)
		
		if len(net.top_names[n]) > 0 and n in layer_names:
			m = net.top_names[n][0]
			D = flattenData(new_data[m])
			mean = np.mean(D, axis=0)
			stddev = np.std(D, axis=0)
			if not quiet:
				print( bcolors.BOLD, ' '*5, n, ':', m, ' '*5, bcolors.ENDC )
				print( 'mean  ', coloredNumbers(mean, bcolors.OKGREEN, bcolors=bcolors) )
				print( 'stddev', coloredNumbers(stddev, bcolors.OKBLUE, bcolors=bcolors) )
				print( 'coef of variation ', bcolors.OKGREEN, stddev.std() / stddev.mean(), bcolors.ENDC )
				print()
			cvar[n] = stddev.std() / stddev.mean()
		# Delete all unused data
		for k in list(active_data):
			if k not in last_used or last_used[k] == i:
				del active_data[k]
	return cvar

def main():
	from argparse import ArgumentParser
	from os import path
	
	parser = ArgumentParser()
	parser.add_argument('prototxt')
	parser.add_argument('-l', '--load', help='Load a caffemodel')
	parser.add_argument('-d', '--data', default=None, help='Image list to use [default prototxt data]')
	#parser.add_argument('-q', action='store_true', help='Quiet execution')
	parser.add_argument('-sm', action='store_true', help='Summary only')
	parser.add_argument('-q', action='store_true', help='Quiet execution')
	parser.add_argument('-a', '--all', action='store_true', help='Show the statistic for all layers')
	parser.add_argument('-nc', action='store_true', help='Do not use color')
	parser.add_argument('-s', type=float, default=1.0, help='Scale the input [only custom data "-d"]')
	parser.add_argument('-bs', type=int, default=16, help='Batch size [only custom data "-d"]')
	parser.add_argument('-nit', type=int, default=10, help='Number of iterations')
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

	if args.load is not None:
		n.copy_from(args.load)
	
	cvar = printMeanStddev(n, NIT=args.nit, show_all=args.all, show_color=not args.nc, quiet=args.sm)
	cv, gr = computeGradientRatio(n, NIT=args.nit)
	print()
	print('  Summary  ')
	print('-----------')
	print()
	print('layer name                         out cvar    rate cvar    rate mean')
	for l in n._layer_names:
		if l in cvar and l in cv and l in gr:
			print('%-30s   %10.2f   %10.2f   %10.2f'%(l, cvar[l], cv[l], gr[l]) )

if __name__ == "__main__":
	main()
