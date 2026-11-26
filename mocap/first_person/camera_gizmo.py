"""World-axis handles for the first frame of a saved camera track."""
import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPen, QPolygonF
from scipy.spatial.transform import Rotation

from ..theme import ui_px

AXES = np.eye(3)
COLORS = ("#f27680", "#98ce75", "#76b1ff")
ANGLES = np.linspace(0., 2*np.pi, 65)


class CameraGizmo:
    def __init__(self, path, stamp, center, saved_alignment):
        self.path, self.stamp = path, stamp
        self.origin = np.asarray(center, dtype=float)
        self.saved_alignment = saved_alignment
        self.transform = np.eye(4)
        self.drag = None
        self.revision = 0

    @property
    def center(self):
        return self.points(self.origin)

    def points(self, points):
        return points @ self.transform[:3, :3].T + self.transform[:3, 3]

    def matrices(self, frames):
        adjusted = frames.copy()
        adjusted[..., :3, :3] = self.transform[:3, :3] @ frames[..., :3, :3]
        adjusted[..., :3, 3] = self.points(frames[..., :3, 3])
        return adjusted

    def reset(self):
        self.drag = None
        self.transform = np.linalg.inv(self.saved_alignment)
        self.revision += 1

    def handles(self, view):
        center = self.center
        xy, depth, focal = view.project(center)
        if depth < .09:
            return
        size = ui_px(61) * depth / focal
        for index in range(3):
            u, v = AXES[(index+1)%3], AXES[(index+2)%3]
            ring = center + size*(np.cos(ANGLES)[:, None]*u + np.sin(ANGLES)[:, None]*v)
            yield "rotate", index, view.project(ring)[0], ANGLES[1]-ANGLES[0]
        for index, axis in enumerate(AXES):
            direction = view.project(center+axis*size)[0]-xy
            length = np.linalg.norm(direction)
            if length >= ui_px(12):
                # Leave screen space between rings and arrows, even on tilted axes.
                radii = np.array([ui_px(82), ui_px(112)])[:, None]
                points = xy + radii*direction/length
                yield "move", index, points, (radii[1, 0]-radii[0, 0])*size/length

    def draw(self, view, painter):
        for kind, index, points, _ in self.handles(view):
            color = QColor(COLORS[index])
            painter.setPen(QPen(color, 2.4))
            painter.drawPolyline(QPolygonF([QPointF(*point) for point in points]))
            if kind == "move":
                direction = (points[-1]-points[0])/np.linalg.norm(points[-1]-points[0])
                side = np.array([-direction[1], direction[0]])
                tip = points[-1]
                base = tip-direction*ui_px(10)
                painter.setBrush(color)
                painter.drawPolygon(QPolygonF([QPointF(*p) for p in
                    (tip, base+side*ui_px(4), base-side*ui_px(4))]))
                painter.drawText(QPointF(*tip)+QPointF(5,-5), "XYZ"[index])

    def hit(self, view, point):
        pixel = np.array([point.x(), point.y()])
        nearest, limit = None, ui_px(10)
        for kind, index, vertices, step in self.handles(view):
            starts = vertices[:-1]
            segments = vertices[1:] - starts
            lengths = np.linalg.norm(segments, axis=1)
            usable = lengths > .1
            direction = segments / np.maximum(.1, lengths[:, None])
            offsets = (pixel-starts)*direction
            foot = starts + np.clip(offsets.sum(axis=1), 0, lengths)[:, None]*direction
            distance = np.where(usable, np.linalg.norm(foot-pixel, axis=1), np.inf)
            edge = np.argmin(distance)
            if distance[edge] < limit:
                limit = distance[edge]
                nearest = (kind, index, direction[edge], step/lengths[edge])
        return nearest

    def begin(self, view, point):
        hit = self.hit(view, point)
        if hit is None:
            return False
        self.drag = (*hit, np.array([point.x(), point.y()]), self.transform.copy())
        return True

    def move(self, point):
        kind, index, direction, scale, start, base = self.drag
        amount = (np.array([point.x(), point.y()])-start) @ direction * scale
        axis = AXES[index]
        self.transform = base.copy()
        if kind == "move":
            self.transform[:3, 3] += axis*amount
        else:
            around = Rotation.from_rotvec(axis*amount).as_matrix()
            center = base[:3, :3] @ self.origin + base[:3, 3]
            self.transform[:3, :3] = around @ base[:3, :3]
            self.transform[:3, 3] = center + around @ (base[:3, 3]-center)
        self.revision += 1
