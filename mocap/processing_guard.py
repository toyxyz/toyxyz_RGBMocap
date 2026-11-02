"""Cheap desktop-side worker supervision; never loads inference libraries."""
import ctypes
import os
import time
from .compute_policy import memory_pressure_message,idle_timeout


class PerformanceInfo(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),*[(key,ctypes.c_size_t) for key in
        ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',
         'KernelTotal','KernelPaged','KernelNonpaged','PageSize')],
        ('HandleCount',ctypes.c_ulong),('ProcessCount',ctypes.c_ulong),('ThreadCount',ctypes.c_ulong)]


def commit_headroom():
    if os.name!='nt':return None
    info=PerformanceInfo();info.cb=ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info),info.cb):return None
    return (info.CommitLimit-info.CommitTotal)*info.PageSize


class ProcessingGuard:
    def __init__(self,pid,now=None,resource_safety=None):
        import psutil
        from .resource_safety import enabled
        self.resource_safety=enabled() if resource_safety is None else bool(resource_safety)
        self.process=psutil.Process(pid)
        self.created=self.process.create_time()
        self.last_output=self.last_cpu_change=time.monotonic() if now is None else now
        self.cpu=0.;self.activity_cpu=0.;self.cpu_samples={};self.low_samples=0

    def output(self):self.last_output=time.monotonic()

    def check(self,now=None):
        import psutil
        now=time.monotonic() if now is None else now
        try:
            if self.process.create_time()!=self.created:return None
            processes=[self.process,*self.process.children(recursive=True)]
            samples={}
            for process in processes:
                try:
                    value=process.cpu_times()
                    key=(process.pid,process.create_time())
                    value=value.user+value.system
                    samples[key]=value
                    self.cpu+=max(0.,value-self.cpu_samples.get(key,0.))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            self.cpu_samples=samples
            if self.cpu>self.activity_cpu+.02:
                self.last_cpu_change=now
                self.activity_cpu=self.cpu
        except psutil.NoSuchProcess:return None
        except psutil.AccessDenied:pass
        if self.resource_safety:
            memory=psutil.virtual_memory();commit=commit_headroom()
            message=memory_pressure_message(memory.available,memory.total,commit)
            low=message is not None
            self.low_samples=self.low_samples+1 if low else 0
            critical=memory.available<256*1024**2 or (commit is not None and commit<256*1024**2)
            if self.low_samples>=2 or critical:return message
        if idle_timeout(self.last_output,self.last_cpu_change,now):
            return 'Processing stopped after 180 seconds without output or CPU activity. The app is still available; retry or use a shorter clip.'
        return None
