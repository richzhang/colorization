import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import cv2


class UserEdit(object):
    def __init__(self, mode, win_size, load_size, img_size):
        self.mode = mode
        self.win_size = win_size
        self.img_size = img_size
        self.load_size = load_size
        print('image_size', self.img_size)
        max_width = np.max(self.img_size)
        self.scale = float(max_width) / self.load_size
        self.dw = int((self.win_size - img_size[0]) / 2)
        self.dh = int((self.win_size - img_size[1]) / 2)
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.ui_count = 0
        print(self)

    def scale_point(self, in_x, in_y, w):
        x = int((in_x - self.dw) / float(self.img_w) * self.load_size) + w
        y = int((in_y - self.dh) / float(self.img_h) * self.load_size) + w
        return x, y

    def __str__(self):
        return "add (%s) with win_size %3.3f, load_size %3.3f" % (self.mode, self.win_size, self.load_size)


class PointEdit(UserEdit):
    def __init__(self, win_size, load_size, img_size):
        UserEdit.__init__(self, 'point', win_size, load_size, img_size)

    def add(self, pnt, color, userColor, width, ui_count):
        self.pnt = pnt
        self.color = color
        self.userColor = userColor
        self.width = width
        self.ui_count = ui_count

    def select_old(self, pnt, ui_count):
        self.pnt = pnt
        self.ui_count = ui_count
        return self.userColor, self.width

    def update_color(self, color, userColor):
        self.color = color
        self.userColor = userColor

    def updateInput(self, im, mask, vis_im):
        w = int(self.width / self.scale)
        pnt = self.pnt
        x1, y1 = self.scale_point(pnt.x(), pnt.y(), -w)
        tl = (x1, y1)
        x2, y2 = self.scale_point(pnt.x(), pnt.y(), w)
        br = (x2, y2)
        c = (self.color.red(), self.color.green(), self.color.blue())
        uc = (self.userColor.red(), self.userColor.green(), self.userColor.blue())
        cv2.rectangle(mask, tl, br, 255, -1)
        cv2.rectangle(im, tl, br, c, -1)
        cv2.rectangle(vis_im, tl, br, uc, -1)

    def is_same(self, pnt):
        dx = abs(self.pnt.x() - pnt.x())
        dy = abs(self.pnt.y() - pnt.y())
        return dx <= self.width+1 and dy <= self.width+1

    def update_painter(self, painter):
        w = max(3, self.width)
        c = self.color
        r = c.red()
        g = c.green()
        b = c.blue()
        ca = QColor(c.red(), c.green(), c.blue(), 255)
        d_to_black = r * r + g * g + b * b
        d_to_white = (255-r) * (255-r) + (255-g) * (255-g) + (255-r) * (255-r)
        if d_to_black > d_to_white:
            painter.setPen(QPen(Qt.black, 1))
        else:
            painter.setPen(QPen(Qt.white, 1))
        painter.setBrush(ca)
        painter.drawRoundedRect(self.pnt.x()-w, self.pnt.y()-w, 1+2*w, 1+2*w, 2, 2)


class UIControl:
    def __init__(self, win_size=256, load_size=512):
        self.win_size = win_size
        self.load_size = load_size
        self.reset()
        self.userEdit = None
        self.userEdits = []
        self.ui_count = 0

    def setImageSize(self, img_size):
        self.img_size = img_size

    def addStroke(self, prevPnt, nextPnt, color, userColor, width):
        pass

    def erasePoint(self, pnt):
        isErase = False
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdits.remove(ue)
                print('remove user edit %d\n' % id)
                isErase = True
                break
        return isErase

    def addPoint(self, pnt, color, userColor, width):
        self.ui_count += 1
        print('process add Point')
        self.userEdit = None
        isNew = True
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdit = ue
                isNew = False
                print('select user edit %d\n' % id)
                break

        if self.userEdit is None:
            self.userEdit = PointEdit(self.win_size, self.load_size, self.img_size)
            self.userEdits.append(self.userEdit)
            print('add user edit %d\n' % len(self.userEdits))
            self.userEdit.add(pnt, color, userColor, width, self.ui_count)
            return userColor, width, isNew
        else:
            userColor, width = self.userEdit.select_old(pnt, self.ui_count)
            return userColor, width, isNew

    def movePoint(self, pnt, color, userColor, width):
        self.userEdit.add(pnt, color, userColor, width, self.ui_count)

    def update_color(self, color, userColor):
        self.userEdit.update_color(color, userColor)

    def update_painter(self, painter):
        for ue in self.userEdits:
            if ue is not None:
                ue.update_painter(painter)

    def get_stroke_image(self, im):
        return im

    def used_colors(self):  # get recently used colors
        if len(self.userEdits) == 0:
            return None
        nEdits = len(self.userEdits)
        ui_counts = np.zeros(nEdits)
        ui_colors = np.zeros((nEdits, 3))
        for n, ue in enumerate(self.userEdits):
            ui_counts[n] = ue.ui_count
            c = ue.userColor
            ui_colors[n, :] = [c.red(), c.green(), c.blue()]

        ui_counts = np.array(ui_counts)
        ids = np.argsort(-ui_counts)
        ui_colors = ui_colors[ids, :]
        unique_colors = []
        for ui_color in ui_colors:
            is_exit = False
            for u_color in unique_colors:
                d = np.sum(np.abs(u_color - ui_color))
                if d < 0.1:
                    is_exit = True
                    break

            if not is_exit:
                unique_colors.append(ui_color)

        unique_colors = np.vstack(unique_colors)
        return unique_colors / 255.0

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)
        vis_im = np.zeros((h, w, 3), np.uint8)

        for ue in self.userEdits:
            ue.updateInput(im, mask, vis_im)
        im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        return im, mask

    def reset(self):
        self.userEdits = []
        self.userEdit = None
        self.ui_count = 0
