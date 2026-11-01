"""Windows shared camera frames. One capture owner, bounded independent readers."""
import ctypes,os,re,struct,time,uuid,math
from multiprocessing import shared_memory
import numpy as np

HEADER=64
MAX_FRAME=64*1024**2
MAX_TOTAL=512*1024**2
FORMAT=struct.Struct('<QdI')

class Mutex:
    def __init__(self,name,create):
        if os.name!='nt':raise RuntimeError('Live shared camera streams require Windows.')
        dll=ctypes.WinDLL('kernel32',use_last_error=True)
        dll.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_wchar_p];dll.CreateMutexW.restype=ctypes.c_void_p
        dll.OpenMutexW.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_wchar_p];dll.OpenMutexW.restype=ctypes.c_void_p
        dll.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];dll.WaitForSingleObject.restype=ctypes.c_ulong
        dll.ReleaseMutex.argtypes=[ctypes.c_void_p];dll.ReleaseMutex.restype=ctypes.c_int
        dll.CloseHandle.argtypes=[ctypes.c_void_p];dll.CloseHandle.restype=ctypes.c_int
        self.dll=dll;self.handle=dll.CreateMutexW(None,False,name) if create else dll.OpenMutexW(0x100001,False,name)
        if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
    def acquire(self,milliseconds=0):
        result=self.dll.WaitForSingleObject(self.handle,milliseconds)
        if result==258:return False
        if result==128:
            self.release();raise RuntimeError('Camera stream owner stopped while updating a frame.')
        if result!=0:raise ctypes.WinError(ctypes.get_last_error())
        return True
    def release(self):self.dll.ReleaseMutex(self.handle)
    def close(self):
        if self.handle:self.dll.CloseHandle(self.handle);self.handle=None

class FrameWriter:
    def __init__(self,shape):
        shape=tuple(int(v) for v in shape)
        if len(shape)!=3 or shape[2]!=3 or min(shape)<1 or max(shape[:2])>8192:raise ValueError('Invalid webcam frame shape')
        self.shape=shape;self.size=math.prod(shape)
        if self.size>MAX_FRAME:raise ValueError('Webcam frame exceeds the shared capture limit; choose a lower capture resolution.')
        self.name='rgbmocap-pear-'+uuid.uuid4().hex;self.memory=None;self.mutex=None;self.pixels=None;self.sequence=0
        try:
            self.mutex=Mutex(self.name+'-lock',True)
            self.memory=shared_memory.SharedMemory(name=self.name,create=True,size=HEADER+self.size)
            self.pixels=np.ndarray(shape,np.uint8,buffer=self.memory.buf,offset=HEADER)
            FORMAT.pack_into(self.memory.buf,0,0,0.,0)
        except Exception:self.close();raise
    def descriptor(self):return dict(name=self.name,shape=list(self.shape),bytes=self.size)
    def publish(self,image,timestamp):
        if image.dtype!=np.uint8 or image.shape!=self.shape:raise ValueError('Webcam frame dimensions changed. Select the camera again.')
        if not math.isfinite(timestamp):raise ValueError('Invalid webcam timestamp')
        if not self.mutex.acquire():return
        try:
            np.copyto(self.pixels,image);self.sequence+=1
            FORMAT.pack_into(self.memory.buf,0,self.sequence,float(timestamp),0)
        finally:self.mutex.release()
    def close(self):
        if self.memory:
            if self.mutex and self.mutex.acquire(50):
                try:FORMAT.pack_into(self.memory.buf,0,self.sequence,time.monotonic(),1)
                finally:self.mutex.release()
            self.pixels=None;self.memory.close();self.memory.unlink();self.memory=None
        if self.mutex:self.mutex.close();self.mutex=None

class RemoteCamera:
    def __init__(self,descriptor):
        name=descriptor['name'];shape=descriptor['shape']
        if not isinstance(name,str) or not re.fullmatch(r'rgbmocap-pear-[0-9a-f]{32}',name):raise ValueError('Invalid shared webcam stream')
        if not isinstance(shape,list) or len(shape)!=3 or any(type(v)!=int or v<1 for v in shape) or shape[2]!=3 or max(shape[:2])>8192:raise ValueError('Invalid shared webcam dimensions')
        size=math.prod(shape)
        if size>MAX_FRAME or descriptor.get('bytes')!=size:raise ValueError('Invalid shared webcam size')
        self.memory=None;self.mutex=None;self.pixels=None;self.sequence=0
        try:
            self.mutex=Mutex(name+'-lock',False);self.memory=shared_memory.SharedMemory(name=name,create=False)
            if not HEADER+size<=self.memory.size<=HEADER+MAX_FRAME+65536:raise ValueError('Shared webcam allocation size changed')
            self.pixels=np.ndarray(tuple(shape),np.uint8,buffer=self.memory.buf,offset=HEADER)
            self.pixels.setflags(write=False)
        except Exception:self.close();raise
    def take(self,timeout=.2):
        deadline=time.monotonic()+max(0.,timeout)
        while True:
            if self.mutex.acquire():
                try:
                    sequence,stamp,closed=FORMAT.unpack_from(self.memory.buf)
                    if closed:raise RuntimeError('Webcam capture was closed. Select the device again.')
                    if sequence and (not math.isfinite(stamp) or stamp>time.monotonic()+1):raise RuntimeError('Invalid webcam stream timestamp')
                    if sequence and time.monotonic()-stamp>5:raise RuntimeError('Webcam stopped delivering fresh frames.')
                    if sequence!=self.sequence and sequence:
                        image=self.pixels.copy();self.sequence=sequence;return image,stamp
                finally:self.mutex.release()
            if time.monotonic()>=deadline:return None
            time.sleep(.003)
    def close(self):
        self.pixels=None
        if self.memory:self.memory.close();self.memory=None
        if self.mutex:self.mutex.close();self.mutex=None

def open_camera(device,height,cv,stream=None):
    if stream is not None:
        if stream.get('device_id')!=device['id']:raise ValueError('Webcam stream does not match the selected device')
        return RemoteCamera(stream)
    from .worker import LatestCamera
    return LatestCamera(device,height,cv)
