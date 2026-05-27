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

## 3. `scripts/run_tworoom_benchmarks.py` — f-string quote fix (Task 10)

**Where:** `scripts/run_tworoom_benchmarks.py`, line ~208 in `render_report()`

**Plan said:**
```python
<div class=\"meta\">Generated {datetime.now().isoformat(timespec=\"seconds\")}. Records: {len(records)}.</div>
```

**This branch shipped:**
```python
<div class=\"meta\">Generated {datetime.now().isoformat(timespec='seconds')}. Records: {len(records)}.</div>
```

**Why:** Backslash-escapes (`\"`) inside an f-string **expression** (`{...}`) are a `SyntaxError` on Python 3.10 (the version this repo targets). They are only legal in the literal text portions of the f-string, where the rest of the HTML's `\"` escapes correctly survive. The fix is the minimum change required for the file to parse: only the one expression was switched to single quotes; every other `\"` in the HTML literal stays.

**If this change breaks something:** It won't — but if you ever upgrade to Python 3.12+, both forms become legal and you could revert if you want consistency with the source fork.

---

## 4. Current swm API differs from the fork's snapshot

The fork was tested against an older `stable-worldmodel`. We validated against `stable-worldmodel==0.1.0` (whatever PyPI shipped as of 2026-05-27) and hit the following deltas that required code/path adjustments **after** initial implementation:

| Surface | Fork assumed | Current swm | Fix applied |
|---|---|---|---|
| HDF5 dataset class | `swm.data.HDF5Dataset` at top level | Only `swm.data.formats.hdf5.HDF5Dataset` (and not auto-registered until the module is imported) | `eval.py` adds `from stable_worldmodel.data.formats.hdf5 import HDF5Dataset` and uses it directly |
| Evaluate method name | `world.evaluate_from_dataset(...)` with `goal_offset_steps=`, `save_video=`, `video_path=` | `world.evaluate(...)` with `goal_offset=`, `video=` | `eval.py` reverted to the older signature: `world.evaluate(..., goal_offset=..., video=results_path)` |
| Dataset file location | `<STABLEWM_HOME>/<name>.h5` | `<STABLEWM_HOME>/datasets/<name>.h5` | README updated; smoke-eval setup moves the file |
| Checkpoint file location | `<STABLEWM_HOME>/<run>/_object.ckpt` | `<STABLEWM_HOME>/checkpoints/<run>/_object.ckpt` | `scripts/convert_hf_tworoom.py` updated to write under `checkpoints/`; `scripts/smoke_tworoom_device.py` updated to read from `checkpoints/` |

These are not security choices — they're forced moves to match the installed swm. If you upgrade or downgrade swm and these symbols/paths shift again, look here first.

## 5. Runtime deps not in `stable-worldmodel[train]`

The `[train]` extra (which we use per the fork to avoid the `gym==0.21.0` build break of `[train,env]` on macOS) does NOT pull in three deps that the runtime actually needs:

- `imageio` — imported by `stable_worldmodel/wrapper/visual.py` at top of swm's `__init__` chain. Without it: `ModuleNotFoundError: No module named 'imageio'` at the `import stable_worldmodel as swm` line in `eval.py`.
- `imageio[ffmpeg]` (i.e. `imageio-ffmpeg`) — needed for mp4 video output during `world.evaluate(..., video=...)`. Without it: `ValueError: Could not find a backend to open ... env_0.mp4`.
- `hdf5plugin` — imported at the top of `stable_worldmodel/data/formats/hdf5.py`. Without it: `ModuleNotFoundError: No module named 'hdf5plugin'` when we try to import the HDF5Dataset class.

The README's macOS section was updated to install all three after the pinned `datasets` / `transformers` step. If swm ever moves them into the `[train]` extra (or you use a different extra that bundles them), the explicit install becomes redundant but not harmful.

---

## How to use this file during RCA

1. **Build/install fails** → check `plan.md` Task 1; this file is unlikely to be the cause.
2. **Smoke script fails on `torch.load`** → check items above for the relevant file.
3. **Eval fails with `float64` errors on MPS** → not a security deviation; check Task 5's `jepa.py` edits.
4. **Eval fails with `cuda` errors on macOS** → not a security deviation; check Task 3 (`resolve_device`) wiring.
5. **Different success rate vs. fork's 86%** → not security-related; check seed, dataset version, planner overrides.

Add new entries to this file whenever a future change deliberately deviates from a reference implementation, so the next debugger doesn't waste time "fixing" the deviation.
