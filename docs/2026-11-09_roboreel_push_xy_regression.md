# 2026-11-09 — RoboReel PushCube-v2 XY/XYZ regression fixed

## Observed server log

The authentic human demonstration was read for episode 0, demonstration 0080, with blue-candidate mask coverage 1.0, in `fixed` mode. **Coverage=1.0 means the color detector produced candidates on every sampled frame, not that the selected object is correct.** Manual inspection of `human_mask_overlay.png` remains required.

Physical rollout reached the `DiagnosticPushTeacher.act` push phase and raised:

```
ValueError: operands could not be broadcast together with shapes (2,) (3,)
agcofl/roboreel_bridge.py, movement=float(np.linalg.norm(cube[:2]-self.last_cube))
```

Root cause: `cube[:2]` is object XY, `self.last_cube` holds XYZ.

## Fix

```python
movement = float(np.linalg.norm(cube[:2] - self.last_cube[:2]))
```

Two regression tests added: proper shape and finite action during push with planar object motion; and stall transition to `recover` after 18 stalled samples. The authoring environment also ran 3 existing MP4/action-interface tests and a direct planar-movement regression successfully. No full physical simulation result has yet been reported following this fix.

## Run *without downloading again*

```bash
cd /mnt/4Tdata/LemonData/cehsisuanfa
git pull origin main
conda activate agcofl4090
unset DISPLAY WAYLAND_DISPLAY
export MPLBACKEND=Agg
python -m pytest -q tests/test_real_video_bridge.py

set -o pipefail
python -u -X faulthandler -m agcofl.roboreel_bridge evaluate \
 --roboreel-root third_party/roboreel --episodes 0 \
 --mode fixed --max-steps 250 --out outputs/roboreel_cn/fixed \
 2>&1 | tee roboreel_cn_fixed_retry.log
cat outputs/roboreel_cn/fixed/report.json
```

Only **after** fixed mode produces `PHYSICAL_RESULT`, run the analogous `--mode video_motion --out outputs/roboreel_cn/video_motion` and `python -m agcofl.analyze_roboreel --out outputs/roboreel_cn`.

Current scientific boundary: real human video loading proven; no claim of successful human-to-robot policy or Full/Anchor/End controlled intervention. The policy remains an oracle-state engineering diagnostic.
