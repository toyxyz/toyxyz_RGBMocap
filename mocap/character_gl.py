"""GPU skinning for imported characters; only a compact bone palette changes per frame."""
import numpy as np
from PySide6.QtOpenGL import QOpenGLShader,QOpenGLBuffer,QOpenGLTexture
from PySide6.QtGui import QImage,QMatrix4x4,QVector4D
from .mesh_gl import MeshRenderer

SKIN='''#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model { float model[]; };
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=4) readonly buffer Frame { float pose[]; };
uniform int vertexCount;
void main(){
 uint i=gl_GlobalInvocationID.x;if(i>=uint(vertexCount))return;
 vec4 v=vec4(model[3*i],model[3*i+1],model[3*i+2],1);
 int r=3*vertexCount+2*int(i);int a=int(model[r]);int n=int(model[r+1]);
 vec3 p=vec3(0);
 for(int k=0;k<n;k++){
  int o=5*vertexCount+2*(a+k);int b=16*int(model[o]);float w=model[o+1];
  p+=w*vec3(dot(vec4(pose[b],pose[b+1],pose[b+2],pose[b+3]),v),
  dot(vec4(pose[b+4],pose[b+5],pose[b+6],pose[b+7]),v),
  dot(vec4(pose[b+8],pose[b+9],pose[b+10],pose[b+11]),v));
 }
 vertices[i].position=vec4(p,1);
}'''
VERTEX='''#version 430 core
layout(location=0) in vec3 position;layout(location=1) in vec3 normal;layout(location=2) in vec3 diffuse;
uniform mat4 mvp;uniform mat4 sceneTransform;out vec3 n;out vec3 c;
void main(){gl_Position=mvp*sceneTransform*vec4(position,1);n=mat3(sceneTransform)*normal;c=diffuse;}'''
FRAGMENT='''#version 430 core
in vec3 n;in vec3 c;out vec4 color;
void main(){vec3 normal=normalize(n);float light=.60+.35*max(0,dot(normal,normalize(vec3(-.4,.8,-.6))));color=vec4(c*light,1);}'''
TEXTURE_VERTEX='''#version 430 core
struct Vertex {vec4 position;vec4 normal;};
layout(std430,binding=1) readonly buffer Output {Vertex vertices[];};
layout(location=0) in vec3 corner;
uniform mat4 mvp;uniform mat4 sceneTransform;out vec3 n;out vec2 uv;
void main(){Vertex v=vertices[int(corner.x)];gl_Position=mvp*sceneTransform*v.position;
n=mat3(sceneTransform)*v.normal.xyz;uv=corner.yz;}'''
TEXTURE_FRAGMENT='''#version 430 core
in vec3 n;in vec2 uv;out vec4 color;
uniform sampler2D baseTexture;uniform bool hasTexture;uniform vec4 baseColor;
void main(){vec4 c=hasTexture?texture(baseTexture,uv):baseColor;
if(c.a<.1)discard;
float light=.60+.35*max(0,dot(normalize(n),normalize(vec3(-.4,.8,-.6))));
color=vec4(c.rgb*light,1);}'''

class CharacterRenderer(MeshRenderer):
    def __init__(self):
        super().__init__()
        if self.cpu:raise RuntimeError('Imported character preview requires OpenGL 4.3; the OpenGL 4.1 fallback is not implemented yet.')
        self.skin.removeAllShaders();self.surface.removeAllShaders()
        self.skin=self.program(((QOpenGLShader.Compute,SKIN),))
        self.surface=self.program(((QOpenGLShader.Vertex,VERTEX),(QOpenGLShader.Fragment,FRAGMENT)))
        self.textured=self.program(((QOpenGLShader.Vertex,TEXTURE_VERTEX),(QOpenGLShader.Fragment,TEXTURE_FRAGMENT)))
        self.textures={};self.batches=[]

    def load(self,asset,palette):
        if self.asset_key==asset['key']:return
        self.release_buffers();self.vertex_count=len(asset['vertices']);self.index_count=asset['faces'].size
        self.vao.bind()
        self.model=self.buffer(np.concatenate([asset['vertices'].ravel(),asset['rows'].ravel(),asset['weights'].ravel()]).astype('float32'))
        self.vertices=self.buffer(self.vertex_count*32)
        self.offsets=self.buffer(asset['offsets']);self.neighbors=self.buffer(asset['neighbors'])
        self.frame=self.buffer(palette*16*4)
        self.indices=self.buffer(asset['faces'],QOpenGLBuffer.IndexBuffer);self.indices.bind()
        self.surface.bind();self.vertices.bind()
        for i,offset in [(0,0),(1,16)]:
            self.surface.enableAttributeArray(i);self.surface.setAttributeBuffer(i,0x1406,offset,3,32)
        self.colors=self.buffer(np.asarray(asset['colors'],np.float32))
        self.surface.enableAttributeArray(2);self.surface.setAttributeBuffer(2,0x1406,0,3,12)
        self.vao.release();self.surface.release()
        try:
            if asset.get('textures'):
                data=asset['textures']
                self.vao.bind();self.textured.bind()
                self.corners=self.buffer(data['corners'])
                self.textured.enableAttributeArray(0);self.textured.setAttributeBuffer(0,0x1406,0,3,12)
                self.vao.release();self.textured.release()
                for name,pixels in data['images'].items():
                    h,w=pixels.shape[:2]
                    limit=int(self.gl.glGetIntegerv(0x0D33))
                    if max(w,h)>limit:raise ValueError(f'Original texture {w}x{h} exceeds this GPU limit ({limit}).')
                    image=QImage(pixels.data,w,h,w*4,QImage.Format_RGBA8888)
                    texture=QOpenGLTexture(image,QOpenGLTexture.GenerateMipMaps)
                    self.textures[name]=texture
                    if not texture.isCreated():raise RuntimeError('Cannot allocate original character texture')
                    texture.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
                    texture.setMagnificationFilter(QOpenGLTexture.Linear)
                    texture.setWrapMode(QOpenGLTexture.Repeat)
                    self.gpu_bytes+=sum(max(1,w>>i)*max(1,h>>i)*4 for i in range(max(w,h).bit_length()))
                self.batches=data['batches']
            self.check('character textures')
        except Exception:
            self.release_buffers();raise
        self.asset_key=asset['key'];self.frame_key=None
        self.check('character load')

    def draw(self,eye,basis,width,height,color=(.71,.71,.71),transform=None,clear_depth=True):
        if not self.batches:
            return super().draw(eye,basis,width,height,color,transform,clear_depth)
        projection=QMatrix4x4();projection.perspective(40,width/max(1,height),.03,2000)
        view=np.eye(4);view[:3,:3]=basis*np.array([[1],[1],[-1]]);view[:3,3]=-view[:3,:3]@eye
        shader=self.textured;gl=self.gl;shader.bind()
        shader.setUniformValue('mvp',projection*QMatrix4x4(*view.ravel().tolist()))
        shader.setUniformValue('sceneTransform',QMatrix4x4(*np.asarray(transform if transform is not None else np.eye(4)).ravel().tolist()))
        gl.glUniform1i(shader.uniformLocation('baseTexture'),0)
        gl.glBindBufferBase(0x90D2,1,self.vertices.bufferId())
        gl.glEnable(0x0B71);gl.glDepthMask(True);gl.glDepthFunc(0x0201)
        for flag in (0x0B90,0x0C11,0x0B44,0x0BE2):gl.glDisable(flag)
        if clear_depth:gl.glClear(0x100)
        self.vao.bind()
        for batch in self.batches:
            texture=self.textures.get(batch['image'])
            gl.glUniform1i(shader.uniformLocation('hasTexture'),int(texture is not None))
            shader.setUniformValue('baseColor',QVector4D(*batch['color']))
            if texture:texture.bind(0)
            gl.glDrawArrays(0x4,batch['start'],batch['count'])
        gl.glBindTexture(0x0DE1,0)
        self.vao.release();shader.release();gl.glDisable(0x0B71)

    def release_buffers(self):
        for texture in self.textures.values():texture.destroy()
        self.textures.clear();self.batches=[]
        super().release_buffers()

    def close(self):
        super().close();self.textured.removeAllShaders()
