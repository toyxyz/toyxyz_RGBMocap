"""Prepare the pinned ViDiHand2 source for the installed FlowHMR runtime."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mocap.first_person.runtime import SETTINGS, SOURCE_REVISION, SOURCE_URL, load_chumpy, source_path, source_ready
from mocap.runtime_registry import activate
from tools.session_lock import app_lock
from tools.source_launch import verify


def verify_source(source):
    if not source_ready(source):
        raise RuntimeError("Incomplete ViDiHand2 source.")
    sys.path.insert(0, str(source))
    try:
        import torch
        from vidihand_v2.inference import load_model

        if not callable(load_model) or not torch.cuda.is_available():
            raise RuntimeError("ViDiHand2 needs a functioning NVIDIA CUDA runtime.")
        q = torch.ones((1, 1, 8, 32), device="cuda", dtype=torch.bfloat16)
        if not torch.nn.functional.scaled_dot_product_attention(q, q, q).isfinite().all().item():
            raise RuntimeError("The CUDA attention check failed.")
    finally:
        sys.path.pop(0)


def install():
    verify(ROOT)
    activate(ROOT, "flow")
    load_chumpy()
    destination = source_path(ROOT)
    if destination.exists():
        verify_source(destination)
        print("ViDiHand2 source already verified.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".install-", dir=destination.parent) as folder:
            checkout = Path(folder) / "checkout"
            checkout.mkdir()
            git = ["git", "-C", str(checkout)]
            subprocess.run([*git, "init", "-q"], check=True, timeout=30)
            subprocess.run([*git, "fetch", "--depth=1", "--no-tags", SOURCE_URL, SOURCE_REVISION],
                           check=True, timeout=180)
            fetched = subprocess.check_output([*git, "rev-parse", "FETCH_HEAD"], text=True, timeout=30).strip()
            if fetched != SOURCE_REVISION:
                raise RuntimeError("Unexpected ViDiHand2 source revision.")
            entries = ("vidihand_v2", "utils", "README.md")
            subprocess.run([*git, "checkout", SOURCE_REVISION, "--", *entries], check=True, timeout=60)
            verify_source(checkout)
            prepared = Path(folder) / "source"
            prepared.mkdir()
            for name in entries:
                (checkout / name).rename(prepared / name)
            prepared.rename(destination)
        print("ViDiHand2 source ready. Prepare the checkpoints in Model Manager.")
    if SETTINGS.is_file():
        print("Existing ViDiHand2 runtime settings are preserved. See docs/FIRST_PERSON_ARMS.md to switch.")


if __name__ == "__main__":
    try:
        with app_lock(ROOT):
            install()
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(f"ViDiHand2 setup failed: {error}")
