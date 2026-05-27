# Apple Silicon MPS Branch — Deliberate Deviations from the Fork

**Branch:** `worktree-apple-silicon-mps`
**Reference fork:** https://github.com/carlosap78/le-wm (commit `56892f9`)
**Plan:** [`../plans/2026-05-27-apple-silicon-mps.md`](../plans/2026-05-27-apple-silicon-mps.md)

This file records places where this branch intentionally **diverges** from the upstream fork it was ported from. If the build fails or behavior differs from the fork's documented results, **check this file first** — the difference may be intentional.

---

## 1. `weights.pt` is loaded with `weights_only=True`

**Where:** `scripts/convert_hf_tworoom.py`, `main()`

**Fork does:**
```python
state_dict = torch.load(SRC / "weights.pt", map_location="cpu", weights_only=False)
```

**This branch does:**
```python
state_dict = torch.load(SRC / "weights.pt", map_location="cpu", weights_only=True)
```

**Why:** `weights.pt` is a tensor-only state dict — it has no class pickles to deserialize. `weights_only=True` is the secure default (no arbitrary code execution from pickle) and works on tensor-only files. The fork's `False` is leftover habit, not a requirement.

**Symptom if this change breaks things:** If `torch.load` raises `UnpicklingError: Weights only load failed`, the HF `weights.pt` may have been re-published containing non-tensor objects (custom optimizers, lambdas, etc.). Flip back to `weights_only=False`, but **only after confirming the source repo is trusted** — the file is fetched from `huggingface.co/quentinll/lewm-tworooms`.

---

## 2. `_object.ckpt` keeps `weights_only=False` (with a warning)

**Where:** `scripts/smoke_tworoom_device.py`, `main()`

**Both fork and this branch do:**
```python
model = torch.load(CKPT, map_location="cpu", weights_only=False).eval().to(device)
```

**Why kept:** `_object.ckpt` is a serialized full `nn.Module` (the `JEPA` instance). PyTorch's safe loader (`weights_only=True`) refuses to unpickle the class — loading would crash. There is no secure alternative within the current swm checkpoint format. The right long-term fix is to switch to state-dict-based checkpoints, which is out of scope for this branch.

**Trust boundary:** never `torch.load` an `_object.ckpt` from an untrusted source. This branch only loads ckpts produced by `convert_hf_tworoom.py` (from the trusted `quentinll/...` HF repos) or already present under `$STABLEWM_HOME`. A new note has been added to the smoke script header to remind future readers.

---

## How to use this file during RCA

1. **Build/install fails** → check `plan.md` Task 1; this file is unlikely to be the cause.
2. **Smoke script fails on `torch.load`** → check items above for the relevant file.
3. **Eval fails with `float64` errors on MPS** → not a security deviation; check Task 5's `jepa.py` edits.
4. **Eval fails with `cuda` errors on macOS** → not a security deviation; check Task 3 (`resolve_device`) wiring.
5. **Different success rate vs. fork's 86%** → not security-related; check seed, dataset version, planner overrides.

Add new entries to this file whenever a future change deliberately deviates from a reference implementation, so the next debugger doesn't waste time "fixing" the deviation.
