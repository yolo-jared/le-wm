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
  <div class=\"meta\">Generated {datetime.now().isoformat(timespec='seconds')}. Records: {len(records)}.</div>
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
