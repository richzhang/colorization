 
import numpy as np
import itertools
import time
import datetime

def check_value(inds, val):
	# Check to see if an array is a single element equaling a particular value
	# Good for pre-processing inputs in a function
	if(np.array(inds).size==1):
		if(inds==val):
			return True
	return False

def flatten_nd_array(pts_nd,axis=1):
	# Flatten an nd array into a 2d array with a certain axis
	# INPUTS
	# 	pts_nd 		N0xN1x...xNd array
	# 	axis 		integer
	# OUTPUTS
	# 	pts_flt 	prod(N \ N_axis) x N_axis array
	NDIM = pts_nd.ndim
	SHP = np.array(pts_nd.shape)
	nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
	NPTS = np.prod(SHP[nax])
	axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
	pts_flt = pts_nd.transpose((axorder))
	pts_flt = pts_flt.reshape(NPTS,SHP[axis])
	return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
	# Unflatten a 2d array with a certain axis
	# INPUTS
	# 	pts_flt 	prod(N \ N_axis) x M array
	# 	pts_nd 		N0xN1x...xNd array
	# 	axis 		integer
	# 	squeeze 	bool 	if true, M=1, squeeze it out
	# OUTPUTS
	# 	pts_out 	N0xN1x...xNd array	
	NDIM = pts_nd.ndim
	SHP = np.array(pts_nd.shape)
	nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
	NPTS = np.prod(SHP[nax])

	if(squeeze):
		axorder = nax
		axorder_rev = np.argsort(axorder)
		M = pts_flt.shape[1]
		NEW_SHP = SHP[nax].tolist()
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

def na():
	return np.newaxis

class Timer():
	def __init__(self):
		self.cur_t = time.time()

	def tic(self):
		self.cur_t = time.time()

	def toc(self):
		return time.time()-self.cur_t

	def tocStr(self, t=-1):
		if(t==-1):
			return str(datetime.timedelta(seconds=np.round(time.time()-self.cur_t,3)))[:-4]
		else:
			return str(datetime.timedelta(seconds=np.round(t,3)))[:-4]

