#!/usr/bin/env bash
# Read-only state of the *current* original official 20×3 Vid2Robot eval.
# Safe from a second terminal: does not touch model, rollout or result files.
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import json
from pathlib import Path

base=Path("outputs/v130_official_vid2robot")
print("=== Official Vid2Robot 20 demos × 3 seed passes ===")
pass_done=0
done_cases=0
known_successes=0
for seed in (0,100000,200000):
    stage=base/f"seed_offset_{seed}"
    progress=stage/"progress.json"
    original=stage/"summary.json"
    if original.is_file():
        try:
            data=json.loads(original.read_text())
            trials=data.get("per_trial",{}).get("demo_distract=False|env_distract=False",[])
            valid=(len(trials)==20 and
                sorted(x.get("demo_id") for x in trials)==list(range(80,100)) and
                all(type(x.get("success")) is bool for x in trials))
            if valid:
                wins=sum(x["success"] for x in trials)
                print(f"seed={seed}: 20/20 physical trials saved, success={wins}/20 "
                      "(await original summary validator if not PASS_VERIFIED)")
                pass_done+=1;done_cases+=20;known_successes+=wins
                continue
            print(f"seed={seed}: summary present but incomplete/invalid; DO NOT COUNT as 20")
        except (OSError,KeyError,TypeError,ValueError) as e:
            print(f"seed={seed}: unreadable original summary: {e}")
    if progress.is_file():
        try:
            d=json.loads(progress.read_text())
            n=int(d.get("completed",0))
            wins=int(d.get("successes_so_far",0))
            if not (0<=n<=20 and 0<=wins<=n):
                raise ValueError("invalid trial counter")
            active=d.get("active_demo_id")
            status=d.get("status")
            print(f"seed={seed}: {n}/20 ({100*n/20:.0f}%) "
                  f"successes_so_far={wins}, current_demo={active}, status={status}")
            done_cases+=n;known_successes+=wins
            continue
        except (OSError,KeyError,TypeError,ValueError) as e:
            print(f"seed={seed}: progress telemetry unreadable: {e}")
    if (stage/"run.log").exists():
        print(f"seed={seed}: pass started; old code has no per-demo telemetry; "
              "use original per-pass completion marker")
    else:
        print(f"seed={seed}: not started")
print(f"Verified or live-counted progress: {done_cases}/60, "
      f"known successes={known_successes} "
      "(partial; no full-3-pass claim until verified)")
status=base/"run_status.json"
if status.exists():
    try:
        d=json.loads(status.read_text())
        print("LAST SAVED RUN_STATUS:",
              d.get("status"),"passes_completed=",d.get("passes_completed"))
    except (ValueError,OSError):
        print("LAST SAVED RUN_STATUS: unreadable")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
 echo "=== NVIDIA current instantaneous sampling ==="
 nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader
else
 echo "nvidia-smi not available; CPU/demo counters are still valid."
fi
