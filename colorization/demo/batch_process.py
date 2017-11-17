#!/usr/bin/python


import os
import sys
import argparse
import numpy as np
from skimage import color, io
import scipy.ndimage.interpolation as sni
import caffe


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('flist', type=str,
                        help='file containing list of images to process')
    parser.add_argument('output', type=str,
                        help='output directory')
    parser.add_argument('-p', '--proto', type=str,
                        default='../models/colorization_deploy_v2.prototxt',
                        help='prototxt file of the net model')
    parser.add_argument('-m', '--model', type=str,
                        default='../models/colorization_release_v2.caffemodel',
                        help='caffemodel file of the net model')
    parser.add_argument('-c', '--cluster', type=str,
                        default='../resources/pts_in_hull.npy',
                        help='cluster centers (pts in hull)')
    parser.add_argument('-g', '--gpu', type=int,
                        default=0,
                        help='gpu id')

    args = parser.parse_args(args=argv)
    return args


# Prepare network
def prepare_net(proto, model, cluster):
    net = caffe.Net(proto, model, caffe.TEST)

    in_shape = net.blobs['data_l'].data.shape[2:] # get input shape
    out_shape = net.blobs['class8_ab'].data.shape[2:] # get output shape

    print 'Input dimensions: %s' % str(in_shape)
    print 'Output dimensions: %s' % str(out_shape)

    pts_in_hull = np.load(cluster) # load cluster centers
    net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel
    print 'Annealed-Mean Parameters populated'

    return net, in_shape, out_shape


# Prepare image for network
def prepare_img(fpath, in_shape):

    # load the original image
    img_rgb = caffe.io.load_image(fpath)

    img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
    img_l = img_lab[:,:,0] # pull out L channel
    orig_shape = img_rgb.shape[:2] # original image size

    # resize image to network input size
    img_rs = caffe.io.resize_image(img_rgb, in_shape) # resize image to network input size
    img_lab_rs = color.rgb2lab(img_rs)
    img_l_rs = img_lab_rs[:,:,0]

    return img_l_rs, img_l, orig_shape


# Process image
def process(net, in_shape, out_shape, fpath):

    img_l_rs, img_l, orig_shape = prepare_img(fpath, in_shape)

    net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
    net.forward() # run network

    ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
    shape = (1.*orig_shape[0]/out_shape[0], 1.*orig_shape[1]/out_shape[1])
    ab_dec_us = sni.zoom(ab_dec,(shape[0],shape[1],1)) # upsample to match size of original image L
    img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
    img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) # convert back to rgb

    return img_rgb_out


# Save image
def save_img(img, fpath, out_dir):
    fname_in = os.path.basename(fpath)
    fpath_out = os.path.join(out_dir, fname_in)

    io.imsave(fpath_out, img)


# Main
def main(argv):

    # Parse arguments
    args = parse_args(argv)
    print args

    # Prepare caffe and net
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    net, in_shape, out_shape = prepare_net(args.proto, args.model, args.cluster)

    # Process files
    with open(args.flist) as flist:
        for fpath in flist:
            fpath = fpath.rstrip('\n')
            print 'Processing file %s ...' % fpath
            img = process(net, in_shape, out_shape, fpath)
            save_img(img, fpath, args.output)

    print 'Done!'


if __name__ == "__main__":
    main(sys.argv[1:])

