
import sys
import argparse
import caffe
from caffe import score, surgery # score, surgery function from caffe-fcn
import numpy as np
import os
import warnings

print sys.argv

def parse_args():
    parser = argparse.ArgumentParser(description='')

    # ***** FLAGS *****
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--phase', dest='phase', help='{0: 0-50k iters, 1: 50-100k, 2: 100-150k}', type=int, default=0)
    parser.add_argument('--caffemodel',dest='caffemodel',help='path to caffemodel', type=str, \
        default='./models/alexnet_release_450000_nobn_rs.caffemodel') # no strokes

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    args = parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu)

    EXP_DIR = os.path.join('./feature_learning_tests/segmentation')
    weights = args.caffemodel

    # make models directory
    # os.chdir(EXP_DIR)
    if(not os.path.exists(os.path.join(EXP_DIR,'models'))):
        print('Making models directory')
        os.mkdir(os.path.join(EXP_DIR,'models'))
    save_format = os.path.join(EXP_DIR,'out_{}')

    if(args.phase==0):
        restore = None
        solver = caffe.SGDSolver(os.path.join(EXP_DIR,'solver0.prototxt'))
    elif(args.phase==1):
        restore = os.path.join(EXP_DIR,'models','fcn_iter_50000.solverstate')
        solver = caffe.SGDSolver(os.path.join(EXP_DIR,'solver1.prototxt'))
    elif(args.phase==2):
        restore = os.path.join(EXP_DIR,'models','fcn_iter_100000.solverstate')
        solver = caffe.SGDSolver(os.path.join(EXP_DIR,'solver2.prototxt'))

    # resume = False
    if restore is not None:
        solver.restore(restore)
    # elif resume:
        # solver.net.copy_from(weights)
    else: 
        solver.net.copy_from(weights) # initialize with weights

        # add bilinear upsampling weights
        interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
        surgery.interp(solver.net, interp_layers)

    net = solver.net
    tnet = solver.test_nets[0]
    solver.test_nets[0].share_with(solver.net)

    LAYER_SCORE = 'score'
    LAYER_LOSS = 'loss'

    # warnings.filterwarnings("ignore")

    # scoring
    val = np.loadtxt(os.path.join(EXP_DIR,'./segvalid11.txt'), dtype=str)
    for aa in range(50000):
        # if(np.mod(aa,100)==0):
            # print 'Running: %i'%aa
        if(np.mod(aa,1000)==0):
            print 'Evaluating: %i'%aa
            score.seg_tests(solver, save_format, val, layer=LAYER_SCORE)
        solver.step(1)
