#!/usr/bin/env bash
# Recover a PARTIALLY INITIALIZED submodule without erasing it or replacing the
# superproject's pinned gitlink. Used ONLY after git submodule update fails.
set -euo pipefail
if (( $# != 4 )); then
  echo "B06_RECOVER_USAGE: target pinned_sha author_git_url entrypoint" >&2
  exit 2
fi
TARGET="$1"
SHA="$2"
AUTHOR_URL="$3"
MARKER="$4"

if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "B06_RECOVER_BLOCKED: invalid pinned SHA" >&2; exit 2
fi
if [[ ! -d "$TARGET" ]]; then
  echo "B06_RECOVER_BLOCKED: target submodule worktree missing" >&2; exit 2
fi
# A parent superproject can misleadingly report --is-inside-work-tree=true.
# Require that this directory is itself the root of an independent Git repo.
TOP="$(git -C "$TARGET" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$TOP" || "$(realpath -e "$TOP")" != "$(realpath -e "$TARGET")" ]]; then
  echo "B06_RECOVER_BLOCKED: target is not an independent Git working tree" >&2
  exit 2
fi

# Never discard user edits/untracked files. An unborn HEAD with an EMPTY
# working tree is safe: fetch the exact gitlink object then detach at that SHA.
if [[ -n "$(git -C "$TARGET" status --porcelain --untracked-files=normal)" ]]; then
  echo "B06_RECOVER_BLOCKED: submodule has local modifications/untracked files" >&2
  exit 2
fi
HEAD_NOW="$(git -C "$TARGET" rev-parse HEAD 2>/dev/null || true)"
if [[ "$HEAD_NOW" == "$SHA" && -s "$TARGET/$MARKER" ]]; then
  echo "B06_RECOVER_READY already_correct sha=$SHA"
  exit 0
fi
if [[ -n "$HEAD_NOW" && "$HEAD_NOW" != "$SHA" ]]; then
  echo "B06_RECOVER_BLOCKED: submodule HEAD points to another commit" >&2
  exit 2
fi

echo "B06_RECOVER_FETCH pinned_sha=$SHA"
# GitHub permits direct object-by-SHA fetch for the referenced original-author
# commit. No change to .gitmodules, remotes, or root repo index.
if ! git -C "$TARGET" fetch --no-tags "$AUTHOR_URL" "$SHA"; then
  # Some remotes disallow fetching unadvertised objects by SHA; fetch the
  # author's named main branch without changing the checked-out working tree.
  echo "B06_RECOVER_FETCH_RETRY published-main-ref"
  git -C "$TARGET" fetch --no-tags "$AUTHOR_URL" \
    '+refs/heads/main:refs/remotes/b06-author/main'
fi
if ! git -C "$TARGET" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  echo "B06_RECOVER_BLOCKED: referenced commit not available after fetch" >&2
  exit 2
fi
git -C "$TARGET" -c advice.detachedHead=false checkout --detach "$SHA"
if [[ "$(git -C "$TARGET" rev-parse HEAD)" != "$SHA" || ! -s "$TARGET/$MARKER" ]]; then
  echo "B06_RECOVER_BLOCKED: pinned checkout lacks required entrypoint" >&2
  exit 2
fi
echo "B06_RECOVER_READY fetched sha=$SHA"
