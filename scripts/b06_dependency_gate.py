#!/usr/bin/env python3
"""B06-only dependency gate; never declare a global pip dependency tree healthy.

The pinned UniSkill-Policy pyproject advertises torch 2.0.1 / torchvision
0.15.2 / transformers 4.36.2. RoboReel ENVIRONMENTS.md intentionally
uses torch 2.10 / torchvision 0.25 / transformers 5.3 for this benchmark.

All metadata warnings originating from the legacy uniskill-policy package are
logged rather than treated as a global resolver failure: RoboReel explicitly
provides a modern, independent runtime and lazy-imports unrelated benchmarks.
The critical torch/vision/transformers versions are checked independently.
The unused decord package is optional: official B06 videos use imageio-ffmpeg.
All OTHER pip-check failures remain blocking. No package metadata is changed.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import re
import subprocess
import sys

EXPECTED = {
    ("torch", "2.0.1"),
    ("torchvision", "0.15.2"),
    ("transformers", "4.36.2"),
}
REGEX = re.compile(
    r"^uniskill-policy 0\.1\.0 has requirement "
    r"(torch|torchvision|transformers)==([0-9.]+), "
    r"but you have [^\s]+\.$"
)

def classify_pip_check(text: str, exit_code: int) -> tuple[list[str], list[str]]:
    tolerated=[]
    blocked=[]
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line=="No broken requirements found.":
            continue
        match=REGEX.fullmatch(line)
        if match and tuple(match.groups()) in EXPECTED:
            tolerated.append(line)
        elif line.startswith("uniskill-policy 0.1.0 "):
            # The original author's packaging declares old dependencies plus
            # optional LIBERO / RoboCasa imports not used by RoboReel.
            # Runtime package pins are checked independently in gate().
            tolerated.append(line)
        elif line == "decord 0.6.0 is not supported on this platform":
            # Optional and not imported by the pinned RoboReel B06 path.
            tolerated.append(line)
        else:
            blocked.append(line)
    if exit_code != 0 and not tolerated and not blocked:
        blocked.append("PIP_CHECK_FAILED_WITHOUT_DIAGNOSTIC")
    return tolerated,blocked

def gate() -> int:
    required={
        "torch": "2.10.0",
        "torchvision": "0.25.0",
        "transformers": "5.3.0",
    }
    for package,want in required.items():
        try:
            installed=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            print("B06_DEPENDENCY_BLOCKED: missing "+package,flush=True)
            return 2
        if installed.split("+")[0] != want:
            print("B06_DEPENDENCY_BLOCKED: incompatible "+package+
                  " actual="+installed+" expected="+want,flush=True)
            return 2
    proc=subprocess.run([sys.executable,"-m","pip","check"],
                        text=True,capture_output=True,check=False)
    allow,blocked=classify_pip_check(proc.stdout+"\n"+proc.stderr,proc.returncode)
    for item in allow:
        print("B06_UPSTREAM_METADATA_CONFLICT_ACKNOWLEDGED "+item,flush=True)
    for item in blocked:
        print("B06_DEPENDENCY_BLOCKED "+item,flush=True)
    if blocked:
        return 2
    print("B06_DEPENDENCY_GATE_OK strict_core_versions; "
          "upstream_metadata_exceptions="+str(len(allow)),flush=True)
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return gate()

if __name__=="__main__":
    raise SystemExit(main())
