"""Bounded FP32 detector batches and exact BGR crops; no TensorRT dependency.

Uses the existing pinned Ultralytics model/operators and PEAR preprocessing.
One graph per current batch shape, at most four images; CPU work uses four threads.
"""
import ast,copy,types,sys
from pathlib import Path
from .core import torch,np,cv2,CROP
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
import torchvision
WARM_STREAM=None
def warm_stream():
 global WARM_STREAM
 if WARM_STREAM is None:WARM_STREAM=torch.cuda.Stream()
 return WARM_STREAM

def hoisted_method(head,name):
 """Only replace two literal CUDA constants; preserve all numerical operations."""
 module=sys.modules[type(head).__module__];tree=ast.parse(Path(module.__file__).read_text('utf8'));cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==type(head).__name__);fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
 class Constants(ast.NodeTransformer):
  def visit_Call(self,node):
   if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch' and node.func.attr=='tensor' and node.args and isinstance(node.args[0],ast.List):
    return ast.copy_location(ast.Attribute(ast.Name('self',ast.Load()),'_graph_R' if isinstance(node.args[0].elts[0],ast.List) else '_graph_bias',ast.Load()),node)
   return self.generic_visit(node)
 fn=Constants().visit(fn);scope=dict(module.__dict__);exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<PEAR fixed CUDA constants>','exec'),scope);return types.MethodType(scope[name],head)

class Net(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.static=None;self.graph=None;self.output=None
    def body(self,x):return self.base.head(self.base.backbone(x).float())
    def forward(self,x):
        x=self.base.normalize(x)[:,:,:,32:-32].half().contiguous()
        if self.static is not None and self.static.shape!=x.shape:self.static=None;self.graph=None;self.output=None
        if self.graph is None:
            self.static=torch.empty_like(x);self.static.copy_(x);stream=warm_stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.body(self.static)
            torch.cuda.current_stream().wait_stream(stream);self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=self.body(self.static)
        self.static.copy_(x);self.graph.replay();return self.output


def leaves(out):
 return [('camera',out['pd_cam'])]+[(group+'_'+k,v) for group in ('body_param','flame_param') for k,v in out[group].items() if torch.is_tensor(v)]

def cpu_output(out,packed=False):
 items=leaves(out)
 if not packed:return {k:v.detach().float().cpu().numpy().copy() for k,v in items}
 sizes=[v.numel() for _,v in items];flat=torch.cat([v.float().reshape(-1) for _,v in items]).cpu().numpy();result={};start=0
 for (k,v),size in zip(items,sizes):result[k]=flat[start:start+size].reshape(tuple(v.shape)).copy();start+=size
 return result
def select(boxes,previous):
 if not len(boxes):return None
 if previous is None:return max(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1])).copy()
 a=np.maximum(0,np.minimum(boxes[:,2:],previous[2:])-np.maximum(boxes[:,:2],previous[:2])).prod(1)
 iou=a/np.maximum(1,(boxes[:,2:]-boxes[:,:2]).prod(1)+(previous[2:]-previous[:2]).prod()-a)
 return boxes[iou.argmax()].copy() if iou.max()>=.1 else None

def crop(im,box,bgr=False,nocopy=False):
 x,y,x2,y2=box
 bbox=CROP['process_bbox'](np.array([x,y,x2-x,y2-y]),im.shape[1],im.shape[0],[256,256],ratio=1.25)
 if bbox is None:return None
 rgb=im if bgr else cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 if nocopy:
  x0=float(bbox[0]+.5*bbox[2]);y0=float(bbox[1]+.5*bbox[3])
  trans=CROP['gen_trans_from_patch_cv'](x0,y0,float(bbox[2]),float(bbox[3]),256,256,1.,0.)
  patch=cv2.warpAffine(rgb,trans,(256,256),flags=cv2.INTER_LINEAR).astype(np.float32)
 else:patch,_,_=CROP['generate_patch_image'](rgb,bbox,1.,0.,False,[256,256])
 if bgr:patch=patch[...,::-1]
 return patch.transpose(2,0,1).copy(),bbox.copy()

class BatchRunner:
    def __init__(self,pear,pool):
        self.pear=pear;self.pool=pool;self.previous=[];self.detector_shape=None;self.graph=None
        self.base=pear.model.original
        # Release the previous single-frame encoder graph before taking ownership.
        pear.model.static=None;pear.model.graphed=None
        if hasattr(pear.model,"graph"):pear.model.graph=None
        head=self.base.head
        head._graph_R=torch.tensor([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]],device='cuda')
        head._graph_bias=torch.tensor([0.,0.,1.5],device='cuda')
        head.forward=hoisted_method(head,'forward');head.get_full_proj=hoisted_method(head,'get_full_proj')
        self.net=Net(self.base).eval()
        self.lb=LetterBox([640,640],auto=False,stride=32)
        self.raw=None;self.static=None;self.gout=None
    def reset(self,count):self.previous=[None]*count
    def pre(self,im):return self.lb(image=im)[...,::-1].transpose(2,0,1).copy()
    @torch.inference_mode()
    def detect(self,images):
        if self.raw is None:
            self.pear.detector.predict(images[0],device=0,classes=[0],conf=.5,imgsz=640,verbose=False)
            self.raw=self.pear.detector.predictor.model.model.float().eval()
        # Auto letterbox is exact for a homogeneous batch; square padding for mixed aspects.
        self.lb.auto=self.auto_letterbox
        x=torch.from_numpy(np.stack(list(self.pool.map(self.pre,images)))).cuda().float()/255
        shape=tuple(x.shape)
        if shape!=self.detector_shape:
            self.graph=None;self.gout=None;self.static=x.clone();self.detector_shape=shape
            for _ in range(4):self.raw(x)
            stream=warm_stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.raw(self.static)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.gout=self.raw(self.static)
        self.static.copy_(x);self.graph.replay();pred=self.gout[0] if isinstance(self.gout,(tuple,list)) else self.gout
        p=pred.transpose(1,2);conf,cls=p[:,:,4:].max(-1);mask=(conf>.5)&(cls==0);ix=mask.nonzero()
        coords=ops.xywh2xyxy(p[ix[:,0],ix[:,1],:4]);scores=conf[mask]
        keep=torchvision.ops.batched_nms(coords.float(),scores.float(),ix[:,0],.7)
        result=[]
        for c,im in enumerate(images):
            ids=keep[ix[keep,0]==c][:300];result.append(ops.scale_boxes(x.shape[2:],coords[ids].clone(),im.shape))
        lengths=[len(v) for v in result];allboxes=torch.cat(result).cpu().numpy()
        return list(np.split(allboxes,np.cumsum(lengths)[:-1]))
    @torch.inference_mode()
    def run(self,images):
        if len(self.previous)!=len(images):self.reset(len(images))
        packets=[None]*len(images);batch_size=min(4,len(images))
        self.auto_letterbox=len({im.shape[:2] for im in images})==1
        # GPU batch size stays bounded independently of camera count. Only valid views enter PEAR.
        for start in range(0,len(images),4):
            group=images[start:start+4]
            detector_images=group+[group[-1]]*(batch_size-len(group))
            detections=self.detect(detector_images)[:len(group)]
            selected=[select(boxes,self.previous[start+i]) for i,boxes in enumerate(detections)]
            for i,box in enumerate(selected):self.previous[start+i]=box
            ids=[i for i,b in enumerate(selected) if b is not None]
            crops=list(self.pool.map(lambda i:crop(group[i],selected[i],True,True),ids))
            valid=[(i,c) for i,c in zip(ids,crops) if c is not None]
            ids=[i for i,c in valid];crops=[c for i,c in valid]
            if not ids:continue
            # Keep the graph's batch shape stable through occlusion. Padded rows
            # are never emitted as observations or included in fusion.
            patches=[v[0] for v in crops];patches+= [patches[-1]]*(batch_size-len(patches))
            out=cpu_output(self.net(torch.from_numpy(np.stack(patches)).cuda()/255),True)
            for j,i in enumerate(ids):
                packets[start+i]=dict(body={k[11:]:v[j:j+1] for k,v in out.items() if k.startswith('body_param_')},
                    flame={k[12:]:v[j:j+1] for k,v in out.items() if k.startswith('flame_param_')},
                    camera=out['camera'][j].copy(),bbox=selected[i].copy(),crop_bbox=crops[j][1],image_size=np.array(group[i].shape[1::-1]))
        return packets
