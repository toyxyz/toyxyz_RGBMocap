"""In-front octahedra using the exact exported world transforms."""
from shiboken6 import VoidPtr
from PySide6.QtGui import QVector3D
from PySide6.QtOpenGL import QOpenGLBuffer
from .rig_gl import RigRenderer


class BoneRenderer(RigRenderer):
    def load(self,asset):
        if self.asset_key==asset['key']:return
        super().load(asset)
        self.vao.bind();self.edge_indices=self.buffer(asset['edges'],QOpenGLBuffer.IndexBuffer)
        self.edge_count=asset['edges'].size;self.indices.bind();self.vao.release()

    def draw(self,*args,color=(.40,.64,.70),edge_color=(.36,.88,1.),**kwargs):
        self.gl.glEnable(0x8037);self.gl.glPolygonOffset(1,1)
        try:super().draw(*args,color=color,**kwargs)
        finally:self.gl.glDisable(0x8037)
        self.surface.bind();self.surface.setUniformValue('bodyColor',QVector3D(*edge_color))
        self.gl.glEnable(0x0B71);self.gl.glDepthFunc(0x0203);self.gl.glDepthMask(False)
        self.vao.bind();self.edge_indices.bind()
        self.gl.glDrawElements(0x1,self.edge_count,0x1405,VoidPtr(0))
        self.indices.bind();self.vao.release();self.gl.glDepthMask(True);self.gl.glDisable(0x0B71)
        self.surface.release();self.check('bone overlay')
