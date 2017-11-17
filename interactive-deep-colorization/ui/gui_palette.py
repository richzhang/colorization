from PyQt4.QtCore import *
from PyQt4.QtGui import *
import numpy as np


class GUIPalette(QWidget):
    def __init__(self, grid_sz=(6, 3)):
        QWidget.__init__(self)
        self.color_width = 25
        self.border = 6
        self.win_width = grid_sz[0] * self.color_width + (grid_sz[0]+1) * self.border
        self.win_height = grid_sz[1] * self.color_width + (grid_sz[1]+1) * self.border
        self.setFixedSize(self.win_width, self.win_height)
        self.num_colors = grid_sz[0] * grid_sz[1]
        self.grid_sz = grid_sz
        self.colors = None
        self.color_id = -1
        self.reset()

    def set_colors(self, colors):
        if colors is not None:
            self.colors = (colors[:min(colors.shape[0], self.num_colors), :] * 255).astype(np.uint8)
            self.color_id = -1
            self.update()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(event.rect(), Qt.white)
        if self.colors is not None:
            for n, c in enumerate(self.colors):
                ca = QColor(c[0], c[1], c[2], 255)
                painter.setPen(QPen(Qt.black, 1))
                painter.setBrush(ca)
                grid_x = n % self.grid_sz[0]
                grid_y = (n - grid_x) / self.grid_sz[0]
                x = grid_x * (self.color_width + self.border) + self.border
                y = grid_y * (self.color_width + self.border) + self.border

                if n == self.color_id:
                    painter.drawEllipse(x, y, self.color_width, self.color_width)
                else:
                    painter.drawRoundedRect(x, y, self.color_width, self.color_width, 2, 2)

        painter.end()

    def sizeHint(self):
        return QSize(self.win_width, self.win_height)

    def reset(self):
        self.colors = None
        self.mouseClicked = False
        self.color_id = -1
        self.update()

    def selected_color(self, pos):
        width = self.color_width + self.border
        dx = pos.x() % width
        dy = pos.y() % width
        if dx >= self.border and dy >= self.border:
            x_id = (pos.x() - dx) / width
            y_id = (pos.y() - dy) / width
            color_id = x_id + y_id * self.grid_sz[0]
            return color_id
        else:
            return -1

    def update_ui(self, color_id):
        self.color_id = color_id
        self.update()
        if color_id >= 0:
            color = self.colors[color_id]
            self.emit(SIGNAL('update_color'), color)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # click the point
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)
            self.mouseClicked = True

    def mouseMoveEvent(self, event):
        if self.mouseClicked:
            color_id = self.selected_color(event.pos())
            self.update_ui(color_id)

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False
