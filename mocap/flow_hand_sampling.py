"""Four-frame hand conditioning; body frames and generator context stay intact."""


def sample_indices(frames, enabled):
    import numpy as np
    if isinstance(frames,bool) or not isinstance(frames,int) or frames<1:
        raise ValueError('Hand sampling requires a positive frame count')
    if not isinstance(enabled,bool):raise ValueError('Hand sampling must be on or off')
    return np.unique(np.r_[np.arange(0,frames,4),frames-1]) if enabled else np.arange(frames)


def interpolate_hands(tokens, indices):
    """Fill hand slots only, in bounded CPU blocks with original anchor tokens."""
    import torch
    if tokens.device.type!='cpu' or tokens.dtype!=torch.float32 or tokens.ndim!=2 or tokens.shape[1]!=3072:
        raise ValueError('Hand interpolation requires CPU float32 body/hand conditioning')
    indices=torch.as_tensor(indices,dtype=torch.long)
    n=len(tokens)
    if (not n or indices.ndim!=1 or not len(indices) or indices[0]!=0 or indices[-1]!=n-1
            or (len(indices)>1 and not bool((indices[1:]>indices[:-1]).all()))):
        raise ValueError('Hand interpolation anchors must cover both endpoints in order')
    anchors=tokens[indices,1024:].clone()
    if not torch.isfinite(anchors).all():raise ValueError('Invalid sampled hand features')
    for start in range(0,n,256):
        at=torch.arange(start,min(n,start+256))
        left=(torch.searchsorted(indices,at,right=True)-1).clamp_min(0)
        right=(left+1).clamp_max(len(indices)-1)
        span=indices[right]-indices[left]
        weight=(at-indices[left]).float()/span.clamp_min(1)
        tokens[start:start+len(at),1024:]=anchors[left]*(1-weight[:,None])+anchors[right]*weight[:,None]
    return tokens
