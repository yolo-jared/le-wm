# Apple Silicon (MPS) Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `eval.py` runnable on Apple Silicon (M-series Macs) using PyTorch's MPS backend, validated end-to-end with the TwoRoom checkpoint, by porting the proven changes from the `carlosap78/le-wm` fork.

**Architecture:** Surgical edits to `eval.py` (device selection, macOS-safe MuJoCo backend, current swm API) and `jepa.py` (float64→float32 cast for Metal). Add three helper scripts under `scripts/` for HF→object-ckpt conversion, MPS smoke testing, and benchmark runs. No new dependencies on the existing model graph; no training-path changes (training is unvalidated on MPS in the upstream fork).

**Tech Stack:** Python 3.10, PyTorch (MPS backend), Hydra, `stable-worldmodel`, `stable-pretraining`, `transformers`, `huggingface_hub` CLI (`hf`), `uv`.

**Hardware target:** Apple M2 Max, 96 GB RAM, macOS. The fork validated on M4 / 16 GB / macOS 26.4.1 / PyTorch 2.12 / Python 3.10.20.

**Testing strategy:** This repo has no unit-test infrastructure and the fork doesn't add any. Verification is done via two runtime gates: (a) `scripts/smoke_tworoom_device.py` after the model edits (encode + predict round-trip on MPS), and (b) a 1-episode `eval.py` smoke run after the eval-script edits. A full paper-like run is the final acceptance check. Failure-path enumeration is documented at the top of each code-modifying task per the project's `implementation-quality.md` rule.

**Reference fork:** https://github.com/carlosap78/le-wm (commit `56892f9`)

**Worktree:** `/Users/jaredsisk/le-wm/.claude/worktrees/apple-silicon-mps` on branch `worktree-apple-silicon-mps`.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Modify | `eval.py` | Skip `MUJOCO_GL=egl` on macOS; pick device via `resolve_device()`; load via `AutoCostModel`; call `evaluate_from_dataset` |
| Modify | `jepa.py` | Add `move_tensor_for_model` helper; use it in `get_cost` so float64 tensors don't reach Metal |
| Create | `scripts/convert_hf_tworoom.py` | Build `JEPA` from `config.json`, load `weights.pt`, save `_object.ckpt` (replaces the inline `vit_hf`-based snippet in the README, which depends on a private `spt` helper) |
| Create | `scripts/smoke_tworoom_device.py` | Load converted ckpt, run `encode` + `predict` on random tensors, print device/shape — fast MPS sanity check |
| Create | `scripts/run_tworoom_benchmarks.py` | Run named preset configurations of `eval.py`, parse metrics, write JSON + HTML report |
| Modify | `.gitignore` | Add `.stable-wm/`, `.cache/`, `.uv-cache/`, `results/` so local artifacts don't leak |
| Modify | `README.md` | Add a short macOS / MPS quickstart section pointing to the new scripts |

**Out of scope (explicitly):**
- Training on MPS (the fork did not validate; `SIGReg` works single-GPU which is fine, but throughput is unknown).
- Other envs (`pusht`, `cube`, `reacher`) — the fork only validated `tworoom`. Other configs may work but are not part of acceptance.
- A pytest harness for this repo.

---

## Pre-flight checks (do this once, before Task 1)

- [ ] **Confirm worktree state**

```bash
cd /Users/jaredsisk/le-wm/.claude/worktrees/apple-silicon-mps
git branch --show-current
```
Expected: `worktree-apple-silicon-mps`

- [ ] **Confirm `hf` CLI is installed** (needed for Task 6's checkpoint download)

```bash
which hf || echo "MISSING: install with 'pip install -U huggingface_hub[cli]' (will also be pulled in by Task 1)"
```

- [ ] **Confirm no in-progress changes**

```bash
git status --short
```
Expected: empty.

---

## Task 1: Set up macOS-compatible Python environment

**Files:**
- Create: `.python-version` (optional pin — only if it doesn't conflict with the user's existing one)

**Goal:** Use the `[train]` extra (not `[train,env]`) to avoid the `gym==0.21.0` build failure on macOS, then pin `datasets` and `transformers` to versions that resolve cleanly.

**Failure paths considered:**
1. `[train,env]` extra fails during build (gym 0.21 + macOS). → Use `[train]` only.
2. Unpinned `datasets` / `transformers` pull versions that conflict with `stable-pretraining`. → Pin to fork-validated versions.
3. `uv` cache in `~/.cache/uv` may already have a broken resolve from a prior attempt. → Use a project-local cache dir (`.uv-cache`).
4. Pre-existing `.venv` from a Linux/CUDA install. → Recreate cleanly.

- [ ] **Step 1: Remove any existing virtual env**

```bash
rm -rf .venv
```

- [ ] **Step 2: Create a fresh Python 3.10 venv**

```bash
uv venv --python=3.10
source .venv/bin/activate
python -V
```
Expected: `Python 3.10.x`

- [ ] **Step 3: Install `stable-worldmodel[train]` (NOT `[train,env]`)**

```bash
UV_CACHE_DIR=.uv-cache uv pip install 'stable-worldmodel[train]'
```
Expected: completes without errors. If it complains about a missing `gym` build, you accidentally used `[train,env]`.

- [ ] **Step 4: Pin `datasets` and `transformers`**

```bash
UV_CACHE_DIR=.uv-cache uv pip install 'datasets==2.21.0' 'transformers==4.46.3'
```
Expected: completes; may show "would change version of X" — that's fine.

- [ ] **Step 5: Confirm PyTorch sees MPS**

```bash
python -c "import torch; print('torch=', torch.__version__); print('mps=', torch.backends.mps.is_available())"
```
Expected:
```
torch= 2.x.x
mps= True
```
If `mps= False`, you are likely on an x86 Python or PyTorch wasn't compiled with MPS — stop and reinstall the arm64 build.

- [ ] **Step 6: Confirm `hf` CLI is available** (it ships with `huggingface_hub[cli]`, pulled transitively)

```bash
hf --help | head -5
```
Expected: usage banner.

- [ ] **Step 7: No commit yet** — Python env state is not tracked. Continue to Task 2.

---

## Task 2: Make MuJoCo backend macOS-safe in `eval.py`

**Files:**
- Modify: `eval.py:1-3`

**Goal:** Don't force `MUJOCO_GL=egl` on macOS — `egl` is Linux-only and breaks MuJoCo imports on Darwin.

**Failure paths considered:**
1. Setting `MUJOCO_GL=egl` on macOS → MuJoCo fails to initialize the GL context, crashes at first env reset. → Skip the env var on Darwin so MuJoCo picks its default (CGL/GLFW).
2. Removing the line entirely → breaks Linux/CUDA users who do need `egl` (headless servers). → Guard with `platform.system() != "Darwin"`.
3. `os.environ["MUJOCO_GL"] = "egl"` unconditionally overrides a user-set value. → Switch to `setdefault` so user can override.

- [ ] **Step 1: Edit `eval.py` to gate the `MUJOCO_GL` set on platform**

Replace lines 1-3:
```python
import os

os.environ["MUJOCO_GL"] = "egl"
```
With:
```python
import os
import platform

if platform.system() != "Darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")
```

- [ ] **Step 2: Verify the file still parses**

```bash
python -c "import ast; ast.parse(open('eval.py').read()); print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add eval.py
git commit -m "fix: skip MUJOCO_GL=egl on macOS (egl is Linux-only)"
```

---

## Task 3: Add `resolve_device()` and use it in `eval.py`

**Files:**
- Modify: `eval.py` (add helper near top after imports; replace `model.to("cuda")` in the policy branch)

**Goal:** Let users select device via `+device=mps` (Hydra additive override) instead of hard-coding `cuda`. Fall back gracefully if requested device isn't available.

**Failure paths considered:**
1. `cfg.device` missing → KeyError. → Use `cfg.get("device", cfg.solver.get("device", "auto"))`.
2. User sets `device=cuda` on a Mac → noisy crash. → Resolver falls back to `mps`/`cpu` with a printed warning.
3. Solver is instantiated from Hydra config and may carry its own `device` field. → Set `cfg.solver.device = device` so they stay in sync before `hydra.utils.instantiate(cfg.solver, ...)`.
4. `torch.backends.mps.is_available()` on a build without MPS → AttributeError. → Not a concern in 2.x; safe.

- [ ] **Step 1: Add `resolve_device` function after the `import` block in `eval.py`**

Insert immediately after line 15 (`import stable_worldmodel as swm`):

```python


def resolve_device(requested=None):
    if requested in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        fallback = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"CUDA is not available; using {fallback}.")
        return fallback

    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS is not available; using cpu.")
        return "cpu"

    return requested
```

- [ ] **Step 2: Verify the file still parses**

```bash
python -c "import ast; ast.parse(open('eval.py').read()); print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Note** — the `model.to("cuda")` line will be replaced in Task 4 along with the `AutoCostModel` switch. Skip ahead.

- [ ] **Step 4: Commit**

```bash
git add eval.py
git commit -m "feat: add resolve_device() helper to eval.py"
```

---

## Task 4: Switch eval policy load to `AutoCostModel` + `evaluate_from_dataset`

**Files:**
- Modify: `eval.py:87-97` (policy-load block) and `eval.py:144-152` (the `world.evaluate(...)` call)

**Goal:** Use the current `stable-worldmodel` API (`AutoCostModel`, `evaluate_from_dataset`) and route the model to the resolved device.

**Failure paths considered:**
1. `swm.wm.utils.load_pretrained` may have been removed or moved in the installed swm version → ImportError at eval time. → Use `swm.policy.AutoCostModel(cfg.policy)` which the README already documents.
2. `world.evaluate(dataset=..., goal_offset=..., video=...)` signature has changed → TypeError. → Per fork, the current API is `world.evaluate_from_dataset(dataset=..., goal_offset_steps=..., save_video=..., video_path=...)`.
3. `cfg.solver.device` may not exist in every solver config (CEM vs Adam) → AttributeError. → Set with assignment; OmegaConf creates the key when missing in struct=False mode (which is the default after Hydra compose). If this fails, fall back to `OmegaConf.update(cfg.solver, "device", device, merge=False)` — but try direct assignment first.
4. `swm.policy.RandomPolicy()` path should remain unchanged.

- [ ] **Step 1: Replace the policy-load branch in `eval.py`**

Find this block (currently around lines 87-97):
```python
    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
```

Replace with:
```python
    if policy != "random":
        device = resolve_device(cfg.get("device", cfg.solver.get("device", "auto")))
        cfg.solver.device = device
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
```

- [ ] **Step 2: Replace the `world.evaluate(...)` call**

Find this block (currently around lines 144-152):
```python
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
```

Replace with:
```python
    metrics = world.evaluate_from_dataset(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        save_video=cfg.output.get("save_video", False),
        video_path=results_path,
    )
```

- [ ] **Step 3: Verify the file still parses**

```bash
python -c "import ast; ast.parse(open('eval.py').read()); print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add eval.py
git commit -m "feat: switch eval.py to AutoCostModel + evaluate_from_dataset + device resolution"
```

---

## Task 5: Add MPS float64 protection in `jepa.py`

**Files:**
- Modify: `jepa.py:8-9` (helper area), `jepa.py:128-137` (`get_cost`)

**Goal:** Metal does not support `float64`. Some swm code paths may produce float64 tensors (numpy → torch defaults to float64; some solvers emit float64 action candidates). Move them through a casting helper before `.to(device)`.

**Failure paths considered:**
1. Direct `.to("mps")` on a float64 tensor → `RuntimeError: Cannot convert a float64 Tensor to MPS`. → Cast to float32 first.
2. Casting non-tensors (e.g. dict entries that aren't tensors) → AttributeError. → Helper short-circuits non-tensors.
3. Casting integer tensors (e.g. indices) → would lose semantics. → Helper only casts when `torch.is_floating_point` is True.
4. `action_candidates` arrives as float64 in `get_cost` → same MPS failure. → Route it through the helper.
5. `info["pixels"]` is sometimes uint8 and sometimes float — `encode` already calls `.float()` (line 34), so it's covered.

- [ ] **Step 1: Add `move_tensor_for_model` helper next to `detach_clone`**

Find (lines 8-9):
```python
def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v
```

Replace with:
```python
def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def move_tensor_for_model(v, device):
    if not torch.is_tensor(v):
        return v
    if torch.is_floating_point(v):
        v = v.float()
    return v.to(device)
```

- [ ] **Step 2: Use the helper in `get_cost`**

Find (around lines 133-137):
```python
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
```

Replace with:
```python
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = move_tensor_for_model(info_dict[k], device)

        action_candidates = move_tensor_for_model(action_candidates, device)
```

- [ ] **Step 3: Verify the file still parses**

```bash
python -c "import ast; ast.parse(open('jepa.py').read()); print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add jepa.py
git commit -m "fix: cast float tensors to float32 before MPS (Metal has no float64)"
```

---

## Task 6: Add `scripts/convert_hf_tworoom.py`

**Files:**
- Create: `scripts/convert_hf_tworoom.py`

**Goal:** Convert the HF-hosted `weights.pt + config.json` into a `_object.ckpt` that `eval.py`'s `AutoCostModel` can consume. The README has an inline snippet, but it uses `spt.backbone.utils.vit_hf` which is a private helper that may have moved. Building the `ViTModel` directly from `transformers` (the fork's approach) is more durable.

**Failure paths considered:**
1. `config.json` schema may differ from what the fork expects (the fork was validated for tworoom). → Build the dict explicitly and document the expected keys.
2. `ViTModel` instantiated without `add_pooling_layer=False` → produces a pooled output that doesn't match `output.last_hidden_state` access in `JEPA.encode`. → Set `add_pooling_layer=False`.
3. `interpolate_pos_encoding` is needed at eval time → set on the config instance.
4. The `predictor` config dict has Hydra meta-keys like `_target_` that would fail `ARPredictor.__init__`. → Strip keys starting with `_`.
5. `weights.pt` is a state-dict (only tensors), so use `weights_only=True` — this avoids arbitrary-code-execution via pickle and is the secure default. (The upstream README and the fork's converter both pass `weights_only=False` out of habit; we deliberately tighten this here.)

- [ ] **Step 1: Create the script file**

Write to `scripts/convert_hf_tworoom.py`:

```python
"""Convert the Hugging Face TwoRoom weights to the object checkpoint format."""

import json
import os
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / ".stable-wm" / "hf_tworooms"
OUT = ROOT / ".stable-wm" / "tworoom" / "lewm_object.ckpt"
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from transformers import ViTConfig, ViTModel

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def build_model(cfg):
    enc = cfg["encoder"]
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    }
    vit_cfg = size_configs[enc["size"]].copy()
    vit_cfg["intermediate_size"] = vit_cfg["hidden_size"] * 4
    vit_cfg["image_size"] = enc["image_size"]
    vit_cfg["patch_size"] = enc["patch_size"]

    encoder = ViTModel(
        ViTConfig(**vit_cfg),
        add_pooling_layer=False,
        use_mask_token=enc.get("use_mask_token", False),
    )
    encoder.config.interpolate_pos_encoding = True

    pred = cfg["predictor"]
    act = cfg["action_encoder"]
    projector = cfg["projector"]
    pred_proj = cfg["pred_proj"]

    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(**without_hydra_keys(pred)),
        action_encoder=Embedder(input_dim=act["input_dim"], emb_dim=act["emb_dim"]),
        projector=MLP(
            input_dim=projector["input_dim"],
            output_dim=projector["output_dim"],
            hidden_dim=projector["hidden_dim"],
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=MLP(
            input_dim=pred_proj["input_dim"],
            output_dim=pred_proj["output_dim"],
            hidden_dim=pred_proj["hidden_dim"],
            norm_fn=nn.BatchNorm1d,
        ),
    )


def without_hydra_keys(values):
    return {key: value for key, value in values.items() if not key.startswith("_")}


def main():
    cfg = json.loads((SRC / "config.json").read_text())
    model = build_model(cfg)
    state_dict = torch.load(SRC / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses**

```bash
python -c "import ast; ast.parse(open('scripts/convert_hf_tworoom.py').read()); print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/convert_hf_tworoom.py
git commit -m "feat: add HF tworoom checkpoint conversion script"
```

---

## Task 7: Add `scripts/smoke_tworoom_device.py` and verify MPS round-trip

**Files:**
- Create: `scripts/smoke_tworoom_device.py`

**Goal:** Fast (~seconds) check that the converted checkpoint loads and that `encode` + `predict` succeed on the selected device. This is the verification gate for Tasks 2–6.

**Failure paths considered:**
1. `lewm_object.ckpt` is missing because Task 9's download hasn't run yet. → Script will FileNotFoundError; that's the expected failure if data isn't in place. The script is harmless to commit before data is downloaded.
2. `torch.load(..., weights_only=False)` is REQUIRED for `_object.ckpt` because it's a serialized full `nn.Module` (not a state dict); `weights_only=True` would refuse to unpickle the `JEPA` class and raise. **Security note:** this means loading any untrusted `_object.ckpt` can execute arbitrary code. We accept this only for checkpoints we converted ourselves (Task 6) or that come from the upstream `quentinll/...` HF repos. Never `torch.load` an `_object.ckpt` from an untrusted source.
3. Random pixel/action tensor shapes must match what `encode` expects: `pixels` is `(B, T, C, H, W)`, `action` is `(B, T, action_dim)`. For tworoom, `action_dim=10` per the fork.
4. Encoder's positional embedding may not match 224×224 if the saved encoder was trained at a different size. → `encoder.config.interpolate_pos_encoding = True` (set in Task 6's converter) handles it.

- [ ] **Step 1: Create the script file**

Write to `scripts/smoke_tworoom_device.py`:

```python
"""Smoke test the converted TwoRoom checkpoint on MPS, CUDA, or CPU."""

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
```

- [ ] **Step 2: Verify the script parses**

```bash
python -c "import ast; ast.parse(open('scripts/smoke_tworoom_device.py').read()); print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_tworoom_device.py
git commit -m "feat: add MPS/CUDA/CPU smoke test for tworoom checkpoint"
```

- [ ] **Step 4: VERIFICATION GATE — defer running until Task 8 downloads the checkpoint.** Move on.

---

## Task 8: Update `.gitignore` for local artifacts

**Files:**
- Modify: `.gitignore`

**Goal:** Prevent local checkpoints, caches, and benchmark output from polluting `git status`.

**Failure paths considered:**
1. `.gitignore` may not exist. → `cat -- existing.gitignore; append` would fail; just check first.
2. Entries may already exist (`.venv`, `__pycache__`). → Don't duplicate.

- [ ] **Step 1: Inspect current `.gitignore`**

```bash
cat .gitignore 2>/dev/null || echo "(no .gitignore)"
```

- [ ] **Step 2: Append macOS / MPS / artifact ignores (skip any that already exist)**

Add the following lines to `.gitignore` (deduplicating with existing entries):

```
# macOS/MPS local artifacts
.stable-wm/
.cache/
.uv-cache/
results/
```

- [ ] **Step 3: Verify `git status` is still clean of artifact paths**

```bash
git status --short
```
Expected: only shows the `.gitignore` edit itself.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore local Apple Silicon artifacts (.stable-wm, .cache, .uv-cache, results)"
```

---

## Task 9: End-to-end gate — download data, convert, smoke, eval

**Files:** none (operational task)

**Goal:** Prove the end-to-end pipeline works on the M2 Max. This is the **acceptance test for the entire plan.**

**Failure paths considered:**
1. `hf download` hangs or hits a rate limit → retry or use a token.
2. `tar --zstd` not supported on older `tar` → install `zstd` via Homebrew; macOS `bsdtar` supports it on recent versions.
3. Smoke test reports `device=cpu` → MPS wasn't detected; re-source the venv, recheck `torch.backends.mps.is_available()`.
4. Smoke test reports `mps_available=True` but `device=cpu` → resolver bug; should not happen with the script as written.
5. Eval crashes with `RuntimeError: ... float64 ...` → Task 5's helper isn't being hit; check `get_cost` was actually edited.
6. Eval crashes with `MUJOCO_GL` related error → Task 2's guard isn't right; on macOS the env var should be unset (not "egl").
7. Eval crashes with `module 'swm.wm.utils' has no attribute 'load_pretrained'` → Task 4's switch to `AutoCostModel` didn't take.

- [ ] **Step 1: Set up local cache paths in the shell**

```bash
export STABLEWM_HOME=$PWD/.stable-wm
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME=$PWD/.cache/huggingface
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.cache/matplotlib
```

- [ ] **Step 2: Download the TwoRoom model checkpoint and convert it**

```bash
hf download quentinll/lewm-tworooms --local-dir .stable-wm/hf_tworooms
python scripts/convert_hf_tworoom.py
```
Expected: prints the path `.stable-wm/tworoom/lewm_object.ckpt`. File exists:
```bash
ls -lh .stable-wm/tworoom/lewm_object.ckpt
```

- [ ] **Step 3: Download and extract the TwoRoom dataset**

```bash
hf download quentinll/lewm-tworooms \
    --repo-type dataset \
    --local-dir .stable-wm/hf_tworooms_dataset

tar --zstd -xvf .stable-wm/hf_tworooms_dataset/tworoom.tar.zst -C "$STABLEWM_HOME"
ls -lh "$STABLEWM_HOME/tworoom.h5"
```
Expected: `tworoom.h5` exists at multi-hundred-MB scale.

- [ ] **Step 4: Run the smoke test (MPS round-trip)**

```bash
python scripts/smoke_tworoom_device.py
```
Expected output:
```
torch=2.x.x
mps_available=True
device=mps
emb=(1, 3, 192) mps:0
pred=(1, 3, 192) mps:0
```

- [ ] **Step 5: Run a fast 1-episode `eval.py` smoke (proves Hydra wiring + `evaluate_from_dataset` + cost flow)**

```bash
python eval.py --config-name=tworoom.yaml \
    policy=tworoom/lewm \
    +device=mps \
    solver.device=mps \
    eval.num_eval=1 \
    eval.eval_budget=5 \
    eval.goal_offset_steps=5 \
    plan_config.horizon=1 \
    plan_config.receding_horizon=1 \
    plan_config.action_block=5 \
    solver.num_samples=2 \
    solver.n_steps=1 \
    solver.topk=1
```
Expected: completes without exceptions, prints a metrics dict with a `success_rate` key.

- [ ] **Step 6: Run the full paper-like TwoRoom evaluation**

```bash
python eval.py --config-name=tworoom.yaml \
    policy=tworoom/lewm \
    +device=mps \
    solver.device=mps
```
Expected: wall time well under 13 min on M2 Max (fork hit 13.85 min on M4-10core); success rate near 86–87%.

- [ ] **Step 7: No commit — this task produces local artifacts that `.gitignore` excludes.** Capture the printed metrics in your final report.

---

## Task 10: Add `scripts/run_tworoom_benchmarks.py`

**Files:**
- Create: `scripts/run_tworoom_benchmarks.py`

**Goal:** Provide named eval presets (smoke / sample / quarter_paper / paper_like), record JSON summaries, and emit an HTML report. Nice-to-have for tracking runs over time and for sharing reproducibility evidence.

**Failure paths considered:**
1. Regex-based metrics parsing may break if `stable-worldmodel` changes its print format. → Script is a benchmarking convenience, not load-bearing; a broken parse leaves `metrics: {}` but doesn't crash the eval itself.
2. Subprocess inherits the parent env including `PYTORCH_ENABLE_MPS_FALLBACK`. → Set defaults in `default_env()`.
3. `--device cuda` on a Mac → eval will fall back via `resolve_device()` (Task 3), so no extra handling needed.
4. Output directory `results/tworoom/` is in `.gitignore` (Task 8). → Output stays local.

- [ ] **Step 1: Create the script file**

Write to `scripts/run_tworoom_benchmarks.py`:

```python
"""Run TwoRoom evaluation presets and generate an HTML report."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tworoom"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
REPORT_PATH = RESULTS_DIR / "report.html"


PRESETS = {
    "smoke": {
        "label": "Smoke",
        "description": "Fast sanity check: model, dataset, environment, and planner.",
        "overrides": {
            "eval.num_eval": 1,
            "eval.eval_budget": 5,
            "eval.goal_offset_steps": 5,
            "plan_config.horizon": 1,
            "plan_config.receding_horizon": 1,
            "plan_config.action_block": 5,
            "solver.num_samples": 2,
            "solver.n_steps": 1,
            "solver.topk": 1,
        },
    },
    "sample": {
        "label": "Sample",
        "description": "Small presentation-friendly sample with several parallel cases.",
        "overrides": {
            "eval.num_eval": 5,
            "eval.eval_budget": 25,
            "eval.goal_offset_steps": 25,
            "plan_config.horizon": 3,
            "plan_config.receding_horizon": 3,
            "plan_config.action_block": 5,
            "solver.num_samples": 50,
            "solver.n_steps": 5,
            "solver.topk": 5,
        },
    },
    "quarter_paper": {
        "label": "Quarter Paper",
        "description": "About one quarter of the default evaluation count; planner defaults stay intact.",
        "overrides": {
            "eval.num_eval": 13,
        },
    },
    "paper_like": {
        "label": "Paper-like",
        "description": "Repository default TwoRoom evaluation. Slowest preset.",
        "overrides": {},
    },
}


def default_env(device: str | None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("STABLEWM_HOME", str(ROOT / ".stable-wm"))
    env.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if device:
        env["LEWM_BENCH_DEVICE"] = device
    return env


def build_command(preset: str, device: str | None) -> list[str]:
    cfg = PRESETS[preset]
    cmd = [
        sys.executable,
        "eval.py",
        "--config-name=tworoom.yaml",
        "policy=tworoom/lewm",
    ]
    if device:
        cmd.extend([f"+device={device}", f"solver.device={device}"])
    for key, value in cfg["overrides"].items():
        cmd.append(f"{key}={value}")
    return cmd


def parse_metrics(output: str) -> dict:
    metrics = {}
    matches = re.findall(r"\{[^\n]*'success_rate'[^\n]*\}", output)
    if matches:
        raw = matches[-1]
        success_match = re.search(r"'success_rate':\s*([0-9.]+)", raw)
        if success_match:
            metrics["success_rate"] = float(success_match.group(1))
        successes_match = re.search(r"'episode_successes':\s*array\((\[[^\)]*\])\)", raw)
        if successes_match:
            values = ast.literal_eval(successes_match.group(1).replace("True", "1").replace("False", "0"))
            metrics["episodes"] = len(values)
            metrics["successes"] = int(sum(values))

    cem_times = [float(x) for x in re.findall(r"CEM solve time:\s*([0-9.]+)\s*seconds", output)]
    if cem_times:
        metrics["cem_solve_times_sec"] = cem_times
        metrics["cem_total_sec"] = round(sum(cem_times), 4)
        metrics["cem_avg_sec"] = round(sum(cem_times) / len(cem_times), 4)
        metrics["cem_calls"] = len(cem_times)
    return metrics


def load_summary() -> list[dict]:
    if not SUMMARY_PATH.exists():
        return []
    return json.loads(SUMMARY_PATH.read_text())


def save_summary(records: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def run_preset(preset: str, device: str | None, tag: str | None) -> dict:
    started_at = datetime.now().isoformat(timespec="seconds")
    cmd = build_command(preset, device)
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=default_env(device),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = round(time.perf_counter() - start, 4)
    output = proc.stdout

    record = {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "tag": tag or "",
        "preset": preset,
        "label": PRESETS[preset]["label"],
        "description": PRESETS[preset]["description"],
        "device": device or "auto",
        "started_at": started_at,
        "elapsed_sec": elapsed,
        "returncode": proc.returncode,
        "command": " ".join(cmd),
        "overrides": PRESETS[preset]["overrides"],
        "metrics": parse_metrics(output),
    }

    log_path = RESULTS_DIR / f"{record['id']}-{preset}.log"
    json_path = RESULTS_DIR / f"{record['id']}-{preset}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output)
    record["log_path"] = str(log_path.relative_to(ROOT))
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    record["json_path"] = str(json_path.relative_to(ROOT))
    return record


def render_report(records: list[dict]) -> None:
    data = json.dumps(records, indent=2)
    rows = "\n".join(
        f"<tr data-preset='{r['preset']}'>"
        f"<td>{r['started_at']}</td>"
        f"<td>{r['label']}</td>"
        f"<td>{r['device']}</td>"
        f"<td>{r['returncode']}</td>"
        f"<td>{r['elapsed_sec']:.2f}</td>"
        f"<td>{r.get('metrics', {}).get('success_rate', '')}</td>"
        f"<td>{r.get('metrics', {}).get('cem_avg_sec', '')}</td>"
        f"<td>{r.get('metrics', {}).get('cem_calls', '')}</td>"
        f"<td><a href='../../{r['log_path']}'>log</a></td>"
        f"</tr>"
        for r in records
    )
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>TwoRoom Benchmark Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", sans-serif; margin: 24px; color: #17202a; }}
    h1 {{ font-size: 28px; margin-bottom: 4px; }}
    .meta {{ color: #5d6d7e; margin-bottom: 20px; }}
    .controls {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
    button {{ border: 1px solid #ccd1d1; background: white; padding: 7px 10px; border-radius: 6px; cursor: pointer; }}
    button.active {{ background: #17202a; color: white; }}
    canvas {{ width: 100%; max-width: 980px; height: 280px; border: 1px solid #e5e8e8; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e5e8e8; padding: 8px; text-align: left; }}
    th {{ background: #f8f9f9; position: sticky; top: 0; }}
    .ok {{ color: #1e8449; }}
  </style>
</head>
<body>
  <h1>TwoRoom Benchmark Report</h1>
  <div class=\"meta\">Generated {datetime.now().isoformat(timespec=\"seconds\")}. Records: {len(records)}.</div>
  <div class=\"controls\" id=\"filters\"></div>
  <canvas id=\"chart\" width=\"980\" height=\"280\"></canvas>
  <table>
    <thead>
      <tr><th>Started</th><th>Preset</th><th>Device</th><th>Code</th><th>Total sec</th><th>Success %</th><th>Avg CEM sec</th><th>CEM calls</th><th>Log</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <script>
    const records = {data};
    let active = \"all\";
    const presets = [\"all\", ...new Set(records.map(r => r.preset))];
    const filters = document.getElementById(\"filters\");
    presets.forEach(p => {{
      const b = document.createElement(\"button\");
      b.textContent = p;
      b.onclick = () => {{ active = p; draw(); }};
      b.dataset.preset = p;
      filters.appendChild(b);
    }});
    function filtered() {{ return active === \"all\" ? records : records.filter(r => r.preset === active); }}
    function draw() {{
      document.querySelectorAll(\"button\").forEach(b => b.classList.toggle(\"active\", b.dataset.preset === active));
      document.querySelectorAll(\"tbody tr\").forEach(tr => tr.style.display = active === \"all\" || tr.dataset.preset === active ? \"\" : \"none\");
      const canvas = document.getElementById(\"chart\");
      const ctx = canvas.getContext(\"2d\");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const data = filtered();
      const max = Math.max(1, ...data.map(r => r.elapsed_sec));
      const barW = Math.max(18, Math.floor((canvas.width - 80) / Math.max(1, data.length)) - 8);
      ctx.font = \"12px sans-serif\";
      ctx.fillStyle = \"#17202a\";
      ctx.fillText(\"Total runtime seconds\", 20, 18);
      data.forEach((r, i) => {{
        const x = 50 + i * (barW + 8);
        const h = Math.round((r.elapsed_sec / max) * 210);
        const y = 250 - h;
        ctx.fillStyle = r.returncode === 0 ? \"#2874a6\" : \"#b03a2e\";
        ctx.fillRect(x, y, barW, h);
        ctx.fillStyle = \"#17202a\";
        ctx.save();
        ctx.translate(x, 265);
        ctx.rotate(-0.7);
        ctx.fillText(r.label, 0, 0);
        ctx.restore();
        ctx.fillText(String(r.elapsed_sec.toFixed(1)), x, y - 4);
      }});
    }}
    draw();
  </script>
</body>
</html>
"""
    REPORT_PATH.write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PRESETS),
        help="Preset to run. Repeatable. Defaults to smoke and sample.",
    )
    parser.add_argument("--all", action="store_true", help="Run every preset, including slow ones.")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    parser.add_argument("--tag", default="", help="Optional label stored with each record.")
    parser.add_argument("--list-presets", action="store_true", help="Print available presets and exit.")
    args = parser.parse_args()

    if args.list_presets:
        for name, cfg in PRESETS.items():
            print(f"{name}: {cfg['description']}")
        return 0

    presets = sorted(PRESETS) if args.all else args.preset or ["smoke", "sample"]
    device = None if args.device == "auto" else args.device

    records = load_summary()
    for preset in presets:
        print(f"Running {preset}...")
        record = run_preset(preset, device, args.tag)
        records.append(record)
        save_summary(records)
        render_report(records)
        status = "ok" if record["returncode"] == 0 else "failed"
        success = record["metrics"].get("success_rate", "n/a")
        print(f"{preset}: {status}, elapsed={record['elapsed_sec']}s, success={success}")

    print(f"Report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the script parses**

```bash
python -c "import ast; ast.parse(open('scripts/run_tworoom_benchmarks.py').read()); print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Run the smoke preset to prove the runner works end-to-end**

```bash
python scripts/run_tworoom_benchmarks.py --preset smoke --device mps --tag m2max-mps
```
Expected: `smoke: ok, elapsed=<seconds>, success=<float>`; files appear under `results/tworoom/`.

- [ ] **Step 4: Inspect the generated report**

```bash
open results/tworoom/report.html
```
Expected: an HTML page with a bar chart and a row for the smoke run.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_tworoom_benchmarks.py
git commit -m "feat: add tworoom benchmark runner with JSON + HTML report output"
```

---

## Task 11: Document macOS / MPS quickstart in README

**Files:**
- Modify: `README.md` (insert a new section after "Installation")

**Goal:** Make the macOS path discoverable for future readers. Keep the upstream README intact; add a single new section.

**Failure paths considered:**
1. README rewording could conflict with upstream. → Add an additive section only; don't edit existing prose.
2. Section may grow stale. → Link to the scripts as the source of truth.

- [ ] **Step 1: Read current `README.md` Installation section to find insertion point**

```bash
grep -n "Installation:" README.md
```

- [ ] **Step 2: Insert the macOS section immediately after the existing `uv pip install ...` block in the Installation section**

Add this content (place it directly below the closing triple-backtick of the upstream `Installation` code block, before the `## Data` heading):

````markdown

### macOS / Apple Silicon (MPS)

Apple Silicon Macs require a slightly different install and a few runtime flags. The TwoRoom evaluation has been validated end-to-end on M-series hardware.

```bash
uv venv --python=3.10
source .venv/bin/activate
UV_CACHE_DIR=.uv-cache uv pip install 'stable-worldmodel[train]'           # NOTE: [train], not [train,env]
UV_CACHE_DIR=.uv-cache uv pip install 'datasets==2.21.0' 'transformers==4.46.3'

export STABLEWM_HOME=$PWD/.stable-wm
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME=$PWD/.cache/huggingface
export XDG_CACHE_HOME=$PWD/.cache
export MPLCONFIGDIR=$PWD/.cache/matplotlib
```

Convert an HF checkpoint and smoke-test MPS:

```bash
hf download quentinll/lewm-tworooms --local-dir .stable-wm/hf_tworooms
python scripts/convert_hf_tworoom.py
python scripts/smoke_tworoom_device.py    # expect device=mps
```

Run TwoRoom eval on Metal:

```bash
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm +device=mps solver.device=mps
```

Optional preset-based benchmark runner with HTML report:

```bash
python scripts/run_tworoom_benchmarks.py --preset smoke --preset sample --device mps --tag m2max-mps
```

````

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add macOS / Apple Silicon (MPS) quickstart to README"
```

---

## Task 12: Finalize the branch

**Goal:** Surface the branch for review / merge decisions.

- [ ] **Step 1: Show the commit summary**

```bash
git log --oneline main..HEAD
```
Expected: 8–10 commits, each scoped to a single change.

- [ ] **Step 2: Show the net diff**

```bash
git diff --stat main..HEAD
```
Expected: 2 modified core files (`eval.py`, `jepa.py`), 3 new scripts, `.gitignore`, `README.md`.

- [ ] **Step 3: Pause and ask the user whether to:**
  - Open a PR (`gh pr create`)
  - Merge directly into `main`
  - Leave the branch in the worktree for further work

Do not pick autonomously — pushing or PR creation is shared-state and should be user-approved per the project's risk policy.

---

## Acceptance criteria

| # | Criterion | How to verify |
|---|---|---|
| 1 | `eval.py` doesn't set `MUJOCO_GL=egl` on macOS | `grep -n MUJOCO_GL eval.py` shows the `platform.system() != "Darwin"` guard |
| 2 | `eval.py` accepts `+device=mps` and routes correctly | Smoke eval (Task 9 Step 5) completes |
| 3 | `eval.py` uses `AutoCostModel` and `evaluate_from_dataset` | `grep -n AutoCostModel eval.py` and `grep -n evaluate_from_dataset eval.py` both match |
| 4 | `jepa.py` casts float64 → float32 before MPS | `grep -n move_tensor_for_model jepa.py` shows the helper and its use in `get_cost` |
| 5 | `scripts/smoke_tworoom_device.py` reports `device=mps` and prints non-zero shapes | Task 9 Step 4 output matches |
| 6 | Full TwoRoom eval prints a success rate near 86–87% | Task 9 Step 6 output |
| 7 | `git status` is clean after Task 11 | `git status --short` empty |

---

## Out-of-scope follow-ups (not part of this plan)

- Training on MPS (validate `train.py` end-to-end on the M2 Max; benchmark `SIGReg` cost on Metal).
- Port the same fixes for `pusht`, `cube`, `reacher` (each may have its own env/asset quirks on macOS).
- Add a pytest harness so future changes have automated coverage.
- Wire `LEWM_BENCH_DEVICE` (set by `run_tworoom_benchmarks.py`) into `eval.py` as a device default — currently unused.
