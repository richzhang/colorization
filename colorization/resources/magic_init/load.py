import caffe

def parseProtoString(s):
	from google.protobuf import text_format
	from caffe.proto import caffe_pb2 as pb
	proto_net = pb.NetParameter()
	text_format.Merge(s, proto_net)
	return proto_net


def get_param(l, exclude=set(['top', 'bottom', 'name', 'type'])):
	if not hasattr(l,'ListFields'):
		if hasattr(l,'__delitem__'):
			return list(l)
		return l
	r = dict()
	for f, v in l.ListFields():
		if f.name not in exclude:
			r[f.name] = get_param(v, [])
	return r

class ProtoDesc:
	def __init__(self, prototxt):
		from os import path
		self.prototxt = prototxt
		self.parsed_proto = parseProtoString(open(self.prototxt, 'r').read())
		# Guess the input dimension
		self.input_dim = (3, 227, 227)
		net = self.parsed_proto
		if len(net.input_dim) > 0:
			self.input_dim = net.input_dim[1:]
		else:
			lrs = net.layer
			cs = [l.transform_param.crop_size for l in lrs
				if l.HasField('transform_param')]
			if len(cs):
				self.input_dim = (3, cs[0], cs[0])

	def __call__(self, clip=None, **inputs):
		from caffe import layers as L
		from collections import OrderedDict
		net = self.parsed_proto
		blobs = OrderedDict(inputs)
		for l in net.layer:
			if l.name not in inputs:
				in_place = l.top == l.bottom
				param = get_param(l)
				assert all([b in blobs for b in l.bottom]), "Some bottoms not founds: " + ', '.join([b for b in l.bottom if not b in blobs])
				tops = getattr(L, l.type)(*[blobs[b] for b in l.bottom],
				                          ntop=len(l.top), in_place=in_place,
				                          name=l.name,
				                          **param)
				if len(l.top) <= 1:
					tops = [tops]
				for i, t in enumerate(l.top):
					blobs[t] = tops[i]
			if l.name == clip:
				break
		return list(blobs.values())[-1]


