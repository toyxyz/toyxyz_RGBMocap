"""Apple Silicon: let CUDA-written inference code run on PyTorch MPS.

The models and their vendored sources address the GPU as CUDA. On a Mac without
CUDA, install() redirects device moves (.cuda(), .to('cuda')) to MPS and answers
the torch.cuda memory queries used for batch plans and Resource Safety from
torch.mps. Nothing changes where CUDA exists. There is no CPU fallback here.

Limits: tensors created directly with device='cuda' are not redirected, MPS has
no peak-memory counters (current figures are reported instead), and MPS raises a
plain RuntimeError when memory runs out, so CUDA OOM recovery paths do not run.
"""
import sys

ACTIVE = False
DEVICE = 'cuda'


def detector_device():
    """Ultralytics device argument: the first CUDA device, or MPS when this layer is active."""
    return DEVICE if ACTIVE else 0


def install():
    """Idempotent; a no-op off macOS, with CUDA, or without MPS."""
    global ACTIVE, DEVICE
    if ACTIVE or sys.platform != 'darwin':
        return ACTIVE
    import torch
    if torch.cuda.is_available() or not torch.backends.mps.is_available():
        return False
    import types
    import psutil
    mps = torch.device('mps')

    def fix(value):
        if isinstance(value, str) and value.startswith('cuda'):
            return mps
        if isinstance(value, torch.device) and value.type == 'cuda':
            return mps
        return value

    for cls in (torch.Tensor, torch.nn.Module):
        def to(self, *args, _to=cls.to, **kwargs):
            return _to(self, *map(fix, args), **{key: fix(value) for key, value in kwargs.items()})
        cls.to = to
        cls.cuda = lambda self, *args, **kwargs: self.to(mps)

    # MPS convolution rejects some strided views (the models crop with x[:, :, :, 32:-32]).
    conv_forward = torch.nn.Conv2d._conv_forward
    torch.nn.Conv2d._conv_forward = lambda self, input, weight, bias: conv_forward(self, input.contiguous(), weight, bias)

    # MPS grid_sample has no border padding (RIFE warps with it). Clamping the grid to the
    # outermost pixel centers and sampling with zero padding reads the same edge pixels.
    functional = torch.nn.functional
    grid_sample = functional.grid_sample

    def border_grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
        if padding_mode == 'border' and input.device.type == 'mps':
            if align_corners:
                grid = grid.clamp(-1, 1)
            else:
                limit = 1 - 1 / torch.tensor(input.shape[:1:-1], dtype=grid.dtype, device=grid.device)
                grid = torch.maximum(torch.minimum(grid, limit), -limit)
            padding_mode = 'zeros'
        return grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    functional.grid_sample = border_grid_sample

    fraction = [1.0]

    def total():
        return torch.mps.recommended_max_memory()

    def mem_get_info(*args):
        # Unified memory: other applications' use of system RAM also limits the GPU.
        return max(0, min(total() - torch.mps.driver_allocated_memory(), psutil.virtual_memory().available)), total()

    def set_fraction(value, *args):
        fraction[0] = float(value)
        torch.mps.set_per_process_memory_fraction(max(0.0, min(2.0, fraction[0])))

    cuda = torch.cuda
    cuda.is_available = lambda: True
    cuda.is_initialized = lambda: True
    cuda.device_count = lambda: 1
    cuda.current_device = lambda: 0
    cuda.get_device_name = lambda *args: 'Apple MPS'
    cuda.get_device_capability = lambda *args: (0, 0)
    cuda.get_device_properties = lambda *args: types.SimpleNamespace(name='Apple MPS', total_memory=total())
    cuda.mem_get_info = mem_get_info
    cuda.memory_allocated = lambda *args: torch.mps.current_allocated_memory()
    cuda.memory_reserved = lambda *args: torch.mps.driver_allocated_memory()
    cuda.max_memory_allocated = lambda *args: torch.mps.current_allocated_memory()
    cuda.max_memory_reserved = lambda *args: torch.mps.driver_allocated_memory()
    cuda.reset_peak_memory_stats = lambda *args: None
    cuda.get_per_process_memory_fraction = lambda *args: fraction[0]
    cuda.set_per_process_memory_fraction = set_fraction
    cuda.empty_cache = torch.mps.empty_cache
    cuda.synchronize = lambda *args: torch.mps.synchronize()
    ACTIVE, DEVICE = True, 'mps'
    return True
