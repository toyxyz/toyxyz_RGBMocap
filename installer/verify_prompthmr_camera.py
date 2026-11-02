"""Exercise native LieTorch, scatter and DroidCalib CUDA operations."""
from pathlib import Path
import argparse,json,sys


def verify(folder):
    sys.path.insert(0,str(folder))
    import torch,lietorch,torch_scatter,droid_backends_intr
    torch.set_num_threads(2)
    x=torch.tensor([[.1,.2,.3,.01,.02,.03]],device='cuda',requires_grad=True)
    transform=lietorch.SE3.exp(x);points=torch.tensor([[1.,2.,3.]],device='cuda')
    y=transform.act(points);y.sum().backward()
    if not torch.isfinite(x.grad).all() or not torch.allclose((transform.inv()*transform).matrix(),torch.eye(4,device='cuda')[None],atol=1e-5):raise ValueError('LieTorch CUDA verification failed.')
    result=torch_scatter.scatter_sum(torch.tensor([1.,2.,3.],device='cuda'),torch.tensor([0,1,0],device='cuda'))
    if result.tolist()!=[4.,2.]:raise ValueError('Scatter CUDA verification failed.')
    poses=lietorch.SE3.Identity(2,device='cuda').data;disps=torch.ones((2,8,8),device='cuda')
    intr=torch.tensor([[10.,10.,4.,4.]]*2,device='cuda');ii=torch.tensor([0],device='cuda');jj=torch.tensor([1],device='cuda')
    distance=droid_backends_intr.frame_distance(poses,disps,intr[0],ii,jj,.3,2)
    if not torch.isfinite(distance).all() or distance.abs().max()>.001:raise ValueError('DroidCalib CUDA verification failed.')
    torch.cuda.synchronize()
    return dict(compute_capability=list(torch.cuda.get_device_capability()),torch=torch.__version__,cuda=torch.version.cuda,lietorch_cuda=True,scatter_cuda=True,droid_cuda=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=verify(args.folder);args.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)
