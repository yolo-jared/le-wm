"""Smoke test the converted TwoRoom checkpoint on MPS, CUDA, or CPU.

WARNING: This script calls torch.load(..., weights_only=False) because
_object.ckpt is a serialized nn.Module (not a state dict). Only load
checkpoints from trusted sources — never untrusted user-supplied files.
"""

from pathlib import Path
import os
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / ".stable-wm" / "tworoom" / "lewm_object.ckpt"
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = resolve_device()
    print(f"torch={torch.__version__}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print(f"device={device}")

    model = torch.load(CKPT, map_location="cpu", weights_only=False).eval().to(device)
    with torch.no_grad():
        batch = {
            "pixels": torch.rand(1, 3, 3, 224, 224, device=device),
            "action": torch.rand(1, 3, 10, device=device),
        }
        out = model.encode(batch)
        pred = model.predict(out["emb"], out["act_emb"])

    print(f"emb={tuple(out['emb'].shape)} {out['emb'].device}")
    print(f"pred={tuple(pred.shape)} {pred.device}")


if __name__ == "__main__":
    main()
