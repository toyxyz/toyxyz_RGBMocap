from __future__ import annotations

import platform


def collect() -> dict:
    import psutil
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "ram_gib": round(psutil.virtual_memory().total / 1024**3, 1), "cuda": False}
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        result["cuda"] = torch.cuda.is_available()
        if result["cuda"]:
            result["gpu"] = torch.cuda.get_device_name(0)
            result["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            result["compute_capability"] = list(torch.cuda.get_device_capability(0))
            x = torch.arange(4096, dtype=torch.float32).cuda().reshape(64, 64)
            value = (x @ x.T).sum().item()
            result["cuda_matmul_verified"] = value > 0
            torch.cuda.synchronize()
    except Exception as error:
        result["error"] = str(error)
    return result
