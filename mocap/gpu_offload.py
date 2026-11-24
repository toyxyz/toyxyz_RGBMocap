"""Bounded synchronous transformer-block offload for inference-only workers.

Weights keep their existing dtype. No pinned-memory pool or background copy
threads are created. The owner closes hooks before discarding/replacing a model.
"""


def groups(model):
    import torch
    found = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith(
                ('blocks', 'frame_blocks', 'inter_frame_blocks')) and len(module):
            if not any(name.startswith(parent + '.') for parent, _ in found):
                found.append((name, module))
    if not found:raise RuntimeError('This model does not support the validated Low Memory block layout.')
    return found


class BlockOffload:
    def __init__(self, model, device):
        self.handles = []
        self.model = model
        self.device = device
        self.closed = False
        selected = groups(model)
        removed = []
        try:
            for name, group in selected:
                parent_name, _, key = name.rpartition('.')
                parent = model.get_submodule(parent_name) if parent_name else model
                del parent._modules[key]
                removed.append((parent, key, group))
            # Offloaded blocks are never temporarily uploaded during loading.
            model.to(device)
        finally:
            for parent, key, group in removed:parent._modules[key] = group
        try:
            for _, group in selected:
                for block in group:
                    block.cpu()
                    self.handles.append(block.register_forward_pre_hook(self._before))
                    self.handles.append(block.register_forward_hook(self._after, always_call=True))
        except BaseException:
            self.close()
            raise

    def _before(self, module, args):
        import torch
        if module.training or torch.is_grad_enabled():
            raise RuntimeError('Low Memory block offload requires evaluation without gradients.')
        module.to(self.device)

    def _after(self, module, args, output):
        module.cpu()

    def close(self):
        if self.closed:return
        self.closed = True
        for handle in self.handles:handle.remove()
        self.handles.clear()
        if self.model is not None:self.model.cpu()
        self.model = None


def load_vggt(checkpoint, device, camera_only=True, low_memory=False, *, optimize_camera=False):
    import gc
    import torch
    from vggt_omega.models import VGGTOmega
    # Only the validated camera-only fast path skips disposable initial weights.
    # Low Memory retains copy loading: assign-backed offload increased Windows
    # private allocation in the paired tests. Keep generic depth loading intact.
    meta_assign = camera_only and not low_memory
    if meta_assign:
        with torch.device('meta'):
            model = VGGTOmega(enable_depth=False).eval()
    else:
        model = VGGTOmega(enable_depth=not camera_only).eval()
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if camera_only:state = {k:v for k,v in state.items() if not k.startswith('dense_head.')}
    if meta_assign:model.load_state_dict(state, assign=True)
    else:model.load_state_dict(state)
    if meta_assign and any(t.is_meta for t in (*model.parameters(), *model.buffers())):
        raise RuntimeError('VGGT camera checkpoint left uninitialized tensors. Repair its checkpoint or runtime.')
    del state
    gc.collect()
    if optimize_camera:
        from .vggt_camera_opt import configure
        configure(model)
    if low_memory:model._mocap_offload = BlockOffload(model, device)
    else:model.to(device)
    return model


def release(model):
    manager = getattr(model, '_mocap_offload', None)
    if manager is not None:
        manager.close()
        del model._mocap_offload
