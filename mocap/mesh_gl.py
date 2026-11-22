"""Full-resolution GPU SMPL-X LBS and smooth normals; no Torch in playback.

Below OpenGL 4.3 (macOS stops at 4.1, without compute shaders) the same skinning and
normals are evaluated with NumPy and uploaded as plain vertex data.
"""
import numpy as np
from shiboken6 import VoidPtr
from PySide6.QtGui import QMatrix4x4, QOpenGLContext, QVector3D
from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram,
    QOpenGLFunctions_4_1_Core, QOpenGLFunctions_4_3_Core, QOpenGLVertexArrayObject)

SKIN = """#version 430 core
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Model { float model[]; };
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=4) readonly buffer Frame { float pose[]; };
uniform int vertexCount;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i>=uint(vertexCount)) return;
    int n=vertexCount*3;
    vec3 p=vec3(model[3*i],model[3*i+1],model[3*i+2]);
    for (int k=0;k<20;k++) {
        int o=n+k*n+int(3*i);
        p+=pose[1366+k]*vec3(model[o],model[o+1],model[o+2]);
    }
    for (int k=0;k<486;k++) {
        int o=n*21+k*n+int(3*i);
        p+=pose[880+k]*vec3(model[o],model[o+1],model[o+2]);
    }
    vec4 v=vec4(p,1);
    vec3 result=vec3(0);
    for (int j=0;j<55;j++) {
        float w=model[n*507+int(i)*55+j];
        if (w==0) continue;
        int o=j*16;
        result+=w*vec3(dot(vec4(pose[o],pose[o+1],pose[o+2],pose[o+3]),v),
                       dot(vec4(pose[o+4],pose[o+5],pose[o+6],pose[o+7]),v),
                       dot(vec4(pose[o+8],pose[o+9],pose[o+10],pose[o+11]),v));
    }
    vertices[i].position=vec4(result+vec3(pose[1386],pose[1387],pose[1388]),1);
}
"""
NORMALS = """#version 430 core
layout(local_size_x=64) in;
struct Vertex { vec4 position; vec4 normal; };
layout(std430,binding=1) buffer Output { Vertex vertices[]; };
layout(std430,binding=2) readonly buffer Offsets { uint offsets[]; };
layout(std430,binding=3) readonly buffer Neighbors { uint neighbors[]; };
uniform int vertexCount;
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i>=uint(vertexCount)) return;
    vec3 center=vertices[i].position.xyz;
    vec3 n=vec3(0);
    for (uint a=offsets[i];a<offsets[i+1];a++) {
        vec3 b=vertices[neighbors[2*a]].position.xyz;
        vec3 c=vertices[neighbors[2*a+1]].position.xyz;
        n+=cross(b-center,c-center);
    }
    vertices[i].normal=vec4(normalize(n),0);
}
"""
VERTEX = """#version 430 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
uniform mat4 mvp;
uniform mat4 sceneTransform;
out vec3 surfaceNormal;
void main() { gl_Position=mvp*sceneTransform*vec4(position,1); surfaceNormal=mat3(sceneTransform)*normal; }
"""
FRAGMENT = """#version 430 core
in vec3 surfaceNormal;
out vec4 color;
uniform vec3 bodyColor;
void main() {
    vec3 n=normalize(surfaceNormal);
    float key=max(0,dot(n,normalize(vec3(-.4,.8,-.6))));
    float fill=max(0,dot(n,normalize(vec3(.6,.3,.7))));
    color=vec4(bodyColor*(.40+.55*key+.20*fill),1);
}
"""


def compute_shaders_available():
    context = QOpenGLContext.currentContext()
    version = context.format().version() if context else (0,0)
    return (version[0],version[1]) >= (4,3)


def smooth_normals(positions, offsets, neighbors):
    """NORMALS on the CPU: area-weighted sum over each vertex's triangle fan."""
    owner = np.repeat(np.arange(len(positions)),np.diff(offsets.astype(np.int64)))
    center = positions[owner]
    cross = np.cross(positions[neighbors[0::2]]-center,positions[neighbors[1::2]]-center)
    total = np.stack([np.bincount(owner,weights=cross[:,axis],minlength=len(positions)) for axis in range(3)],axis=1)
    return total/np.maximum(np.linalg.norm(total,axis=1,keepdims=True),1e-20)


class MeshRenderer:
    def __init__(self,shared=None):
        self.cpu = shared.cpu if shared else not compute_shaders_available()
        self.gl = QOpenGLFunctions_4_1_Core() if self.cpu else QOpenGLFunctions_4_3_Core()
        if not self.gl.initializeOpenGLFunctions():
            raise RuntimeError("SMPL-X Mesh requires OpenGL 4.1 or newer. Update the graphics driver.")
        self.shared=shared
        self.skin = shared.skin if shared else self.compute_program(SKIN)
        self.normals = shared.normals if shared else self.compute_program(NORMALS)
        self.surface = shared.surface if shared else self.program(((QOpenGLShader.Vertex,VERTEX),(QOpenGLShader.Fragment,FRAGMENT)))
        self.buffers = []
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.asset_key = None
        self.frame_key = None
        self.vertex_count = self.index_count = self.gpu_bytes = 0
        self.dispatches = 0
        self.driver = self.gl.glGetString(0x1F01)

    def check(self,label):
        error = self.gl.glGetError()
        if error:
            raise RuntimeError(f"OpenGL {error} at {label}")

    def compute_program(self,source):
        return None if self.cpu else self.program(((QOpenGLShader.Compute,source),))

    def replace_compute(self,source):
        if self.skin is not None:self.skin.removeAllShaders()
        self.skin = self.compute_program(source)

    def program(self,stages):
        p = QOpenGLShaderProgram()
        for stage,source in stages:
            if self.cpu:source = source.replace('#version 430 core','#version 410 core',1)
            if not p.addShaderFromSourceCode(stage,source):
                raise RuntimeError(p.log())
        if not p.link():
            raise RuntimeError(p.log())
        return p

    def buffer(self, data, kind=None):
        b = QOpenGLBuffer(kind or QOpenGLBuffer.VertexBuffer)
        if not b.create() or not b.bind():
            raise RuntimeError("Cannot create a mesh GPU buffer")
        if isinstance(data,int):
            b.allocate(data)
            size = data
        else:
            value = data.tobytes()
            b.allocate(value,len(value))
            size = len(value)
        self.buffers.append(b)
        self.gpu_bytes += size
        return b

    def load(self, asset):
        if asset["key"] == self.asset_key:
            return
        self.release_buffers()
        self.vertex_count = int(asset["vertex_count"])
        self.index_count = asset["faces"].size
        self.vao.bind()
        owner = self.shared if self.shared and self.shared.asset_key==asset['key'] else None
        if self.cpu:
            self.cpu_load(asset)
            self.vertices = self.buffer(self.vertex_count*32)
        else:
            self.model = owner.model if owner else self.buffer(asset["packed"])
            self.vertices = self.buffer(self.vertex_count*32)
            self.offsets = owner.offsets if owner else self.buffer(asset["offsets"])
            self.neighbors = owner.neighbors if owner else self.buffer(asset["neighbors"])
            self.frame = self.buffer(1389*4)
        self.indices = owner.indices if owner else self.buffer(asset["faces"],QOpenGLBuffer.IndexBuffer)
        self.indices.bind()
        if not self.cpu:
            for index,b in enumerate((self.model,self.vertices,self.offsets,self.neighbors,self.frame)):
                self.gl.glBindBufferBase(0x90D2,index,b.bufferId())
        self.surface.bind()
        self.vertices.bind()
        self.surface.enableAttributeArray(0)
        self.surface.enableAttributeArray(1)
        self.surface.setAttributeBuffer(0,0x1406,0,3,32)
        self.surface.setAttributeBuffer(1,0x1406,16,3,32)
        self.vao.release()
        self.surface.release()
        self.asset_key = asset["key"]
        self.frame_key = None
        self.check("load")

    def cpu_load(self, asset):
        """Views into the packed model: template, 20 shape and 486 pose directions, 55 weights."""
        count = self.vertex_count; n = count*3
        packed = np.asarray(asset["packed"],np.float32).reshape(-1)
        self.cpu_template = packed[:n].reshape(count,3)
        self.cpu_shape = packed[n:21*n].reshape(20,n)
        self.cpu_pose = packed[21*n:507*n].reshape(486,n)
        self.cpu_weights = packed[507*n:507*n+count*55].reshape(count,55)
        self.cpu_topology(asset)

    def cpu_topology(self, asset):
        self.cpu_offsets = np.asarray(asset["offsets"]).reshape(-1)
        self.cpu_neighbors = np.asarray(asset["neighbors"]).reshape(-1)

    def cpu_skin(self, frame):
        """SKIN on the CPU; returns positions and, when the skin provides them, normals."""
        rest = self.cpu_template+(frame[1366:1386]@self.cpu_shape+frame[880:1366]@self.cpu_pose).reshape(-1,3)
        blend = np.tensordot(self.cpu_weights,frame[:880].reshape(55,4,4)[:,:3],1)
        return np.einsum('vrc,vc->vr',blend[:,:,:3],rest)+blend[:,:,3]+frame[1386:1389],None

    def cpu_upload(self, positions, normals):
        if normals is None:normals = smooth_normals(positions,self.cpu_offsets,self.cpu_neighbors)
        data = np.zeros((self.vertex_count,8),np.float32)
        data[:,:3] = positions; data[:,3] = 1; data[:,4:7] = normals
        self.cpu_vertices = data
        self.vertices.bind()
        self.vertices.write(0,data.tobytes(),data.nbytes)

    def deform(self, frame, key):
        if key == self.frame_key:
            return
        frame = np.ascontiguousarray(frame,dtype=np.float32)
        if self.cpu:
            self.cpu_upload(*self.cpu_skin(frame.reshape(-1)))
            self.check("vertex upload")
            self.frame_key = key
            self.dispatches += 1
            return
        self.frame.bind()
        self.frame.write(0,frame.tobytes(),frame.nbytes)
        self.check("frame upload")
        for index,b in enumerate((self.model,self.vertices,self.offsets,self.neighbors,self.frame)):
            self.gl.glBindBufferBase(0x90D2,index,b.bufferId())
        self.check("bind storage")
        for program in (self.skin,self.normals):
            program.bind()
            self.check("bind program")
            self.gl.glUniform1i(program.uniformLocation("vertexCount"),self.vertex_count)
            self.check("uniform")
            self.gl.glDispatchCompute((self.vertex_count+63)//64,1,1)
            self.check("compute")
            self.gl.glMemoryBarrier(0x2000 | 0x1)
            program.release()
        self.frame_key = key
        self.dispatches += 1

    def draw(self, eye, basis, width, height, color=(.71,.71,.71),transform=None,clear_depth=True):
        projection = QMatrix4x4()
        projection.perspective(40,width/max(1,height),.03,2000)
        view = np.eye(4)
        view[:3,:3] = basis * np.array([[1],[1],[-1]])
        view[:3,3] = -view[:3,:3] @ eye
        self.surface.bind()
        self.surface.setUniformValue("mvp",projection * QMatrix4x4(*view.reshape(-1).tolist()))
        self.surface.setUniformValue('sceneTransform',QMatrix4x4(*np.asarray(transform if transform is not None else np.eye(4)).reshape(-1).tolist()))
        self.surface.setUniformValue('bodyColor',QVector3D(*color))
        self.gl.glEnable(0x0B71)  # Depth test; true surface occlusion, not face sorting.
        self.gl.glDepthMask(True)
        self.gl.glDepthFunc(0x0201)
        self.gl.glDisable(0x0B90)
        self.gl.glDisable(0x0C11)
        self.gl.glDisable(0x0B44)
        if clear_depth:self.gl.glClear(0x100)
        self.vao.bind()
        self.gl.glDrawElements(0x4,self.index_count,0x1405,VoidPtr(0))
        self.vao.release()
        self.gl.glDisable(0x0B44)
        self.gl.glDisable(0x0B71)
        self.surface.release()

    def read_vertices(self):
        # Diagnostic only: a GPU readback would stall normal playback.
        result = np.empty((self.vertex_count,8),np.float32)
        self.vertices.bind()
        if not self.vertices.read(0,result,result.nbytes):
            raise RuntimeError("Cannot read the verification buffer")
        return result

    def release_buffers(self):
        for b in self.buffers:
            b.destroy()
        self.buffers.clear()
        self.asset_key = self.frame_key = None
        self.gpu_bytes = 0

    def close(self):
        self.release_buffers()
        self.vao.destroy()
        if not self.shared:
            for p in (self.skin,self.normals,self.surface):
                if p is not None:p.removeAllShaders()
