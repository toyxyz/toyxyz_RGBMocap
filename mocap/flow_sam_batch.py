"""Measured-throughput SAM batches; no precision or temporal changes."""
GIB = 1024**3
MAX_BATCH = 16


def working_bytes(batch, width, height, gpu_crops):
    # In addition to loaded weights: decoder activations and (when selected)
    # vectorised full-resolution left/right GPU crop inputs. Calibrated at
    # 1080p on RTX 5090; larger images scale their upload/float workspace.
    # At 1080p the decoder peak dominates even with CPU crops. Do not mistake
    # disabling GPU crops for removing those per-frame activations.
    image_extra=max(0,36*(width*height-1920*1080)) if gpu_crops else 0
    return int(.25*GIB + batch*(.13*GIB + image_extra))


def choose_batch_size(frames, width, height, available_gpu, available_ram,
                      *, enabled=True, low_memory=False, gpu_crops=True):
    if not enabled or low_memory:return 1
    # Current + two prepared batches + a crop stack, independent of clip length.
    for batch in (MAX_BATCH, 8, 4, 2):
        if (batch <= frames and working_bytes(batch,width,height,gpu_crops) <= available_gpu
                and 4*batch*width*height*3 <= available_ram):
            return batch
    return 1


def current_batch_size(frames, width, height, *, enabled=True, low_memory=False,
                       gpu_crops=True):
    if not enabled or low_memory:return 1
    import torch,psutil
    from .compute_policy import gpu_reserve,ram_reserve
    free,total=torch.cuda.mem_get_info()
    allocated=torch.cuda.memory_allocated()
    cached=max(0,torch.cuda.memory_reserved()-allocated)
    cap=torch.cuda.get_per_process_memory_fraction()*total
    available=max(0,min(free+cached-gpu_reserve(total),cap-allocated))
    ram=psutil.virtual_memory()
    return choose_batch_size(frames,width,height,available,max(0,ram.available-ram_reserve(ram.total)),
                             enabled=enabled,low_memory=low_memory,gpu_crops=gpu_crops)


def extract_bounded(extractor,images,boxes,intrinsics,*,gpu_crops,placement,limit,emit):
    """Retry only an unpublished sub-batch after releasing the OOM traceback."""
    import torch,gc
    from .flow_fast_sam import extract_batch
    values=[];people=[];offset=0
    while offset<len(images):
        count=min(limit,len(images)-offset)
        failed=False
        try:
            value,output=extract_batch(extractor,images[offset:offset+count],boxes[offset:offset+count],
                                       intrinsics,gpu_crops=gpu_crops,placement=placement)
        except torch.cuda.OutOfMemoryError:
            if count==1:raise
            failed=True
        if failed:
            # The exception and its activation references are gone here.
            gc.collect();torch.cuda.empty_cache()
            limit=max(1,min(limit//2,count//2))
            emit('info',message=f'SAM GPU memory pressure: retrying unpublished frames with batch {limit}.')
            continue
        values.append(value);people.extend(output);offset+=count
    return torch.cat(values),people,limit
