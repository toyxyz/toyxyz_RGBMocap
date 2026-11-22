"""Small canonical actor binding over the shared registered SMPL-X buffers."""
import numpy as np
from PySide6.QtOpenGL import QOpenGLShader
from .mesh_gl import MeshRenderer

SKIN="""#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model { float model[]; };
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=4) readonly buffer Frame { float pose[]; };
layout(std430,binding=5) readonly buffer Binding { float bindData[]; };
uniform int vertexCount;
void main() {
 uint i=gl_GlobalInvocationID.x;
 if(i>=uint(vertexCount)) return;
 int n=vertexCount*3;
 vec3 delta=vec3(0);
 for(int k=0;k<486;k++) {
  int o=n*21+k*n+int(3*i);
  delta+=pose[880+k]*vec3(model[o],model[o+1],model[o+2]);
 }
 vec3 transported=vec3(0);
 for(int j=0;j<55;j++) {
  float w=model[n*507+int(i)*55+j];
  if(w==0) continue;
  int o=n+j*9;
  transported+=w*vec3(dot(vec3(bindData[o],bindData[o+1],bindData[o+2]),delta),
   dot(vec3(bindData[o+3],bindData[o+4],bindData[o+5]),delta),
   dot(vec3(bindData[o+6],bindData[o+7],bindData[o+8]),delta));
 }
 vec4 p=vec4(vec3(bindData[3*i],bindData[3*i+1],bindData[3*i+2])+transported,1);
 vec3 result=vec3(0);
 for(int j=0;j<55;j++) {
  float w=model[n*507+int(i)*55+j];
  if(w==0) continue;
  int o=j*16;
  result+=w*vec3(dot(vec4(pose[o],pose[o+1],pose[o+2],pose[o+3]),p),
   dot(vec4(pose[o+4],pose[o+5],pose[o+6],pose[o+7]),p),
   dot(vec4(pose[o+8],pose[o+9],pose[o+10],pose[o+11]),p));
 }
 vertices[i].position=vec4(result,1);
}
"""


class BoundMeshRenderer(MeshRenderer):
    def __init__(self,shared=None):
        self.binding_buffer=None;self.binding_key=None
        super().__init__(shared)
        if shared is None:self.replace_compute(SKIN)

    def release_buffers(self):
        super().release_buffers()
        self.binding_buffer=None;self.binding_key=None

    def bind_skin(self,binding):
        key=binding.get('key',id(binding))
        if key==self.binding_key:return
        value=np.concatenate((binding['vertices'].reshape(-1),binding['frames'].reshape(-1))).astype(np.float32)
        if self.cpu:
            count=self.vertex_count
            self.cpu_bind_vertices=value[:count*3].reshape(count,3)
            self.cpu_bind_blend=np.tensordot(self.cpu_weights,value[count*3:].reshape(55,3,3),1)
            self.binding_buffer=True;self.binding_key=key;self.frame_key=None;return
        if self.binding_buffer is None:self.binding_buffer=self.buffer(value)
        else:
            self.binding_buffer.bind()
            if self.binding_buffer.size()!=value.nbytes:raise ValueError('Canonical binding dimensions changed')
            self.binding_buffer.write(0,value.tobytes(),value.nbytes)
        self.binding_key=key;self.frame_key=None

    def cpu_skin(self,frame):
        delta=(frame[880:1366]@self.cpu_pose).reshape(-1,3)
        rest=self.cpu_bind_vertices+np.einsum('vrc,vc->vr',self.cpu_bind_blend,delta)
        blend=np.tensordot(self.cpu_weights,frame[:880].reshape(55,4,4)[:,:3],1)
        return np.einsum('vrc,vc->vr',blend[:,:,:3],rest)+blend[:,:,3],None

    def deform(self,frame,key):
        if self.binding_buffer is None:raise ValueError('Canonical skin binding missing')
        if not self.cpu:self.gl.glBindBufferBase(0x90D2,5,self.binding_buffer.bufferId())
        super().deform(frame,(self.binding_key,key))
