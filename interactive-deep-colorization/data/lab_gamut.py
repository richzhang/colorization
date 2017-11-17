
import numpy as np
import scipy as sp
from skimage import color
from pdb import set_trace as st
import warnings


def qcolor2lab_1d(qc):
    # take 1d numpy array and do color conversion
    c = np.array([qc.red(), qc.green(), qc.blue()], np.uint8)
    return rgb2lab_1d(c)


def rgb2lab_1d(in_rgb):
    # take 1d numpy array and do color conversion
    # print('in_rgb', in_rgb)
    return color.rgb2lab(in_rgb[np.newaxis, np.newaxis, :]).flatten()


def lab2rgb_1d(in_lab, clip=True, dtype='uint8'):
    warnings.filterwarnings("ignore")
    tmp_rgb = color.lab2rgb(in_lab[np.newaxis, np.newaxis, :]).flatten()
    if clip:
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
    if dtype == 'uint8':
        tmp_rgb = np.round(tmp_rgb * 255).astype('uint8')
    return tmp_rgb


def snap_ab(input_l, input_rgb, return_type='rgb'):
    ''' given an input lightness and rgb, snap the color into a region where l,a,b is in-gamut
    '''
    T = 20
    warnings.filterwarnings("ignore")
    input_lab = rgb2lab_1d(np.array(input_rgb))  # convert input to lab
    conv_lab = input_lab.copy()  # keep ab from input
    for t in range(T):
        conv_lab[0] = input_l  # overwrite input l with input ab
        old_lab = conv_lab
        tmp_rgb = color.lab2rgb(conv_lab[np.newaxis, np.newaxis, :]).flatten()
        tmp_rgb = np.clip(tmp_rgb, 0, 1)
        conv_lab = color.rgb2lab(tmp_rgb[np.newaxis, np.newaxis, :]).flatten()
        dif_lab = np.sum(np.abs(conv_lab-old_lab))
        if dif_lab < 1:
            break
        # print(conv_lab)

    conv_rgb_ingamut = lab2rgb_1d(conv_lab, clip=True, dtype='uint8')
    if (return_type == 'rgb'):
        return conv_rgb_ingamut

    elif(return_type == 'lab'):
        conv_lab_ingamut = rgb2lab_1d(conv_rgb_ingamut)
        return conv_lab_ingamut


class abGrid():
    def __init__(self, gamut_size=110, D=1):
        self.D = D
        self.vals_b, self.vals_a = np.meshgrid(np.arange(-gamut_size, gamut_size+D, D),
                                               np.arange(-gamut_size, gamut_size+D, D))
        self.pts_full_grid = np.concatenate((self.vals_a[:, :, np.newaxis], self.vals_b[:, :, np.newaxis]), axis=2)
        self.A = self.pts_full_grid.shape[0]
        self.B = self.pts_full_grid.shape[1]
        self.AB = self.A * self.B
        self.gamut_size = gamut_size

    def update_gamut(self, l_in):
        warnings.filterwarnings("ignore")
        thresh = 1.0
        pts_lab = np.concatenate((l_in + np.zeros((self.A, self.B, 1)), self.pts_full_grid), axis=2)
        self.pts_rgb = (255 * np.clip(color.lab2rgb(pts_lab), 0, 1)).astype('uint8')
        pts_lab_back = color.rgb2lab(self.pts_rgb)
        pts_lab_diff = np.linalg.norm(pts_lab-pts_lab_back, axis=2)

        self.mask = pts_lab_diff < thresh
        mask3 = np.tile(self.mask[..., np.newaxis], [1, 1, 3])
        self.masked_rgb = self.pts_rgb.copy()
        self.masked_rgb[np.invert(mask3)] = 255
        return self.masked_rgb, self.mask

    def ab2xy(self, a, b):
        y = self.gamut_size + a
        x = self.gamut_size + b
        # print('ab2xy (%d, %d) -> (%d, %d)' % (a, b, x, y))
        return x, y

    def xy2ab(self, x, y):
        a = y - self.gamut_size
        b = x - self.gamut_size
        # print('xy2ab (%d, %d) -> (%d, %d)' % (x, y, a, b))
        return a, b
