#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

SOURCE_COMMIT="7949df58f6ef302622c9d957b38177c18372bbc8"
REPO_COMMIT="7d0a6243fd8b92500b5f864d70b3b60d6b6c853a"
HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
URL="\${HF_ENDPOINT%/}/YixiangChen/EC-Flow/resolve/\${SOURCE_COMMIT}/flow.pt"
DEST="third_party/EC-Flow/ckpt/flow.pt"
PART="\${DEST}.partial"
SOURCE_FILE="\${PART}.source"
MIN_BYTES=9000000000

[[ -d third_party/EC-Flow/.git ]] || { echo "ERROR: pinned EC-Flow source clone missing"; exit 2; }
ACTUAL_COMMIT="$(git -C third_party/EC-Flow rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$REPO_COMMIT" ]] || { echo "ERROR: EC-Flow source revision mismatch: $ACTUAL_COMMIT"; exit 3; }

mkdir -p "$(dirname "$DEST")"
if [[ -f "$DEST" ]]; then
  SIZE="$(stat -c '%s' "$DEST")"
  if (( SIZE >= MIN_BYTES )); then
    echo "TRO1_B12_FLOW_ALREADY_PRESENT bytes=$SIZE path=$DEST"
    exit 0
  fi
  echo "ERROR: refusing to overwrite existing undersized checkpoint: $DEST ($SIZE bytes)"
  exit 4
fi

if [[ -e "$PART" ]]; then
  [[ -f "$SOURCE_FILE" ]] || { echo "ERROR: partial file has no source lock; refusing resume"; exit 5; }
  [[ "$(cat "$SOURCE_FILE")" == "$URL" ]] || { echo "ERROR: partial file source URL mismatch"; exit 6; }
else
  printf '%s\n' "$URL" > "$SOURCE_FILE"
fi

FREE_BYTES="$(df -PB1 "$(dirname "$DEST")" | awk 'NR==2 {print $4}')"
(( FREE_BYTES >= MIN_BYTES + 1073741824 )) || { echo "ERROR: insufficient disk space"; exit 7; }

echo "TRO1_B12_FLOW_DOWNLOAD_BEGIN source=$URL destination=$DEST"
curl --fail --location --retry 4 --retry-delay 5 --retry-all-errors --continue-at - "$URL" --output "$PART"

SIZE="$(stat -c '%s' "$PART")"
(( SIZE >= MIN_BYTES )) || { echo "ERROR: downloaded file too small ($SIZE bytes); keeping partial"; exit 8; }
mv "$PART" "$DEST"
rm -f "$SOURCE_FILE"
SHA="$(sha256sum "$DEST" | awk '{print $1}')"
python - "$DEST" "$SIZE" "$SHA" "$SOURCE_COMMIT" "$REPO_COMMIT" <<'PY'
import json, pathlib, sys
path, size, sha, model_rev, source_rev = sys.argv[1:]
meta = {
    "baseline": "B12_EC_FLOW",
    "asset": "flow.pt",
    "path": path,
    "bytes": int(size),
    "sha256": sha,
    "model_hub_revision": model_rev,
    "source_repository_revision": source_rev,
    "status": "DOWNLOADED_NOT_LOADED",
}
out = pathlib.Path(path + ".provenance.json")
out.write_text(json.dumps(meta, indent=2) + "\n")
print("TRO1_B12_FLOW_DOWNLOADED", json.dumps(meta))
PY
