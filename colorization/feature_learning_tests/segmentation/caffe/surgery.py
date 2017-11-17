from __future__ import division
import caffe
import numpy as np

def transplant(new_net, net):
    for p in net.params:
        if p not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if net.params[p][i].data.shape != new_net.params[p][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p][i].data.shape
            else:
                print 'copying', p, i
            new_net.params[p][i].data.flat = net.params[p][i].data.flat

def expand_score(new_net, new_layer, net, layer):
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0,0,0,:old_cl][...] = net.params[layer][1].data

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

def interp(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt

def upsample_filt2(size1,size2):
    size = np.maximum(size1,size2)
    factor = (size + 1) // 2
    if size1 % 2 == 1:
        center1 = factor - 1
    else:
        center1 = factor - 0.5

    if size2 % 2 == 1:
        center2 = factor - 1
    else:
        center2 = factor - 0.5

    og = np.ogrid[:size1, :size2]
    return (1 - abs(og[0] - center1) / factor) * \
           (1 - abs(og[1] - center2) / factor)

def interp2(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k and k != 1:
            print 'input + output channels need to be the same or |output| == 1'
            raise
        filt = upsample_filt2(h,w)
        net.params[l][0].data[range(m), range(k), :, :] = filt
