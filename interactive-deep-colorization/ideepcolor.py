from __future__ import print_function
import platform
import sys
import argparse
import qdarkstyle
from PyQt4.QtGui import QApplication, QIcon
from PyQt4.QtCore import Qt
from ui import gui_design
from data import colorize_image as CI

sys.path.append('./caffe_files')


def parse_args():
    parser = argparse.ArgumentParser(description='iDeepColor: deep interactive colorization')
    # basic parameters
    parser.add_argument('--win_size', dest='win_size', help='the size of the main window', type=int, default=512)
    parser.add_argument('--image_file', dest='image_file', help='input image', type=str, default='test_imgs/mortar_pestle.jpg')
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--cpu_mode', dest='cpu_mode', help='do not use gpu', action='store_true')

    # Main colorization model
    parser.add_argument('--color_prototxt', dest='color_prototxt', help='colorization caffe prototxt', type=str,
                        default='./models/reference_model/deploy_nodist.prototxt')
    parser.add_argument('--color_caffemodel', dest='color_caffemodel', help='colorization caffe prototxt', type=str,
                        default='./models/reference_model/model.caffemodel')

    # Distribution prediction model
    parser.add_argument('--dist_prototxt', dest='dist_prototxt', type=str, help='distribution net prototxt',
                        default='./models/reference_model/deploy_nopred.prototxt')
    parser.add_argument('--dist_caffemodel', dest='dist_caffemodel', type=str, help='distribution net caffemodel',
                        default='./models/reference_model/model.caffemodel')

    # ***** DEPRECATED *****
    parser.add_argument('--load_size', dest='load_size', help='image size', type=int, default=256)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    for arg in vars(args):
        print('[%s] =' % arg, getattr(args, arg))

    if(args.cpu_mode):
        args.gpu = -1

    args.win_size = int(args.win_size / 4.0) * 4  # make sure the width of the image can be divided by 4
    # initialize the colorization model
    colorModel = CI.ColorizeImageCaffe(Xd=args.load_size)
    colorModel.prep_net(args.gpu, args.color_prototxt, args.color_caffemodel)

    distModel = CI.ColorizeImageCaffeDist(Xd=args.load_size)
    distModel.prep_net(args.gpu, args.dist_prototxt, args.dist_caffemodel)

    # initialize application
    app = QApplication(sys.argv)
    window = gui_design.GUIDesign(color_model=colorModel, dist_model=distModel,
                                  img_file=args.image_file, load_size=args.load_size, win_size=args.win_size)
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))  # comment this if you do not like dark stylesheet
    app.setWindowIcon(QIcon('imgs/logo.png'))  # load logo
    window.setWindowTitle('iColor')
    window.setWindowFlags(window.windowFlags() & ~Qt.WindowMaximizeButtonHint)   # fix window siz
    window.show()
    app.exec_()
