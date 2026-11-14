#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

clone_pinned() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local dest="$4"

  if [[ -e "$dest" ]]; then
    if [[ ! -d "$dest/.git" ]]; then
      echo "ERROR: destination exists but is not a git clone: $dest" >&2
      return 10
    fi
    local current
    current="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
    local dirty
    dirty="$(git -C "$dest" status --porcelain)"
    local origin
    origin="$(git -C "$dest" remote get-url origin 2>/dev/null || true)"
    if [[ "$current" == "$revision" && -z "$dirty" && "$origin" == "$url" ]]; then
      echo "TRO1_SOURCE_ALREADY_PINNED $name $current"
      return 0
    fi
    echo "ERROR: refusing to alter existing source tree: $dest" >&2
    echo "  current=$current expected=$revision origin=$origin dirty=\${dirty:+yes}" >&2
    return 11
  fi

  mkdir -p "$(dirname "$dest")"
  git init "$dest"
  git -C "$dest" remote add origin "$url"
  git -C "$dest" fetch --filter=blob:none --depth 1 origin "$revision"
  git -C "$dest" checkout --detach FETCH_HEAD

  local actual
  actual="$(git -C "$dest" rev-parse HEAD)"
  [[ "$actual" == "$revision" ]] || {
    echo "ERROR: $name revision mismatch: $actual != $revision" >&2
    return 12
  }
  echo "TRO1_SOURCE_PINNED $name $actual $dest"
}

clone_pinned \
  "B11_Im2Flow2Act" \
  "https://github.com/real-stanford/im2Flow2Act.git" \
  "f44b368b76195c30bf38074f6419fc8b9f8feb5f" \
  "third_party/im2Flow2Act"

clone_pinned \
  "B12_EC-Flow" \
  "https://github.com/YixiangChen515/EC-Flow.git" \
  "7d0a6243fd8b92500b5f864d70b3b60d6b6c853a" \
  "third_party/EC-Flow"

echo
echo "TRO1_B11_B12_PINNED_SOURCE_CLONE_COMPLETE"
echo "No conda environment, pip package, checkpoint, dataset, training, or simulator was touched."
