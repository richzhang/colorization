import cv2
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from data import lab_gamut
import numpy as np


class GUIGamut(QWidget):
    def __init__(self, gamut_size=110):
        QWidget.__init__(self)
        self.gamut_size = gamut_size
        self.win_size = gamut_size * 2  # divided by 4
        self.setFixedSize(self.win_size, self.win_size)
        self.ab_grid = lab_gamut.abGrid(gamut_size=gamut_size, D=1)
        self.reset()

    def set_gamut(self, l_in=50):
        self.l_in = l_in
        self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=l_in)
        self.update()

    def set_ab(self, color):
        self.color = color
        self.lab = lab_gamut.rgb2lab_1d(self.color)
        x, y = self.ab_grid.ab2xy(self.lab[1], self.lab[2])
        self.pos = QPointF(x, y)
        self.update()

    def is_valid_point(self, pos):
        if pos is None:
            return False
        else:
            x = pos.x()
            y = pos.y()
            if x >= 0 and y >= 0 and x < self.win_size and y < self.win_size:
                return self.mask[y, x]
            else:
                return False

    def update_ui(self, pos):
        self.pos = pos
        a, b = self.ab_grid.xy2ab(pos.x(), pos.y())
        # get color we need L
        L = self.l_in
        lab = np.array([L, a, b])
        color = lab_gamut.lab2rgb_1d(lab, clip=True, dtype='uint8')
        self.emit(SIGNAL('update_color'), color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), Qt.white)
        if self.ab_map is not None:
            ab_map = cv2.resize(self.ab_map, (self.win_size, self.win_size))
            qImg = QImage(ab_map.tostring(), self.win_size, self.win_size, QImage.Format_RGB888)
            painter.drawImage(0, 0, qImg)

        painter.setPen(QPen(Qt.gray, 3, Qt.DotLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
        painter.drawLine(self.win_size/2, 0, self.win_size/2, self.win_size)
        painter.drawLine(0, self.win_size/2, self.win_size, self.win_size/2)
        if self.pos is not None:
            painter.setPen(QPen(Qt.black, 2, Qt.SolidLine, cap=Qt.RoundCap, join=Qt.RoundJoin))
            w = 5
            x = self.pos.x()
            y = self.pos.y()
            painter.drawLine(x - w, y, x + w, y)
            painter.drawLine(x, y - w, x, y + w)
        painter.end()

    def mousePressEvent(self, event):
        pos = event.pos()

        if event.button() == Qt.LeftButton and self.is_valid_point(pos):  # click the point
            self.update_ui(pos)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self.is_valid_point(pos):
            if self.mouseClicked:
                self.update_ui(pos)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)

    def reset(self):
        self.ab_map = None
        self.mask = None
        self.color = None
        self.lab = None
        self.pos = None
        self.mouseClicked = False
        self.update()
