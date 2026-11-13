#!/usr/bin/env bash
# Non-destructive exact-gitlink recovery for an interrupted RoboReel submodule.
set -euo pipefail
if (( $# != 4 )); then
  echo 'B06_RECOVER_USAGE: target pinned_sha author_git_url entrypoint' >&2; exit 2
fi
TARGET="$1" SHA="$2" AUTHOR_URL="$3" MARKER="$4"
if [[ ! "$SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'B06_RECOVER_BLOCKED: invalid pinned SHA' >&2; exit 2
fi
if [[ ! -d "$TARGET" || -L "$TARGET" || "$MARKER" = /* || "$MARKER" = *'..'* ]]; then
  echo 'B06_RECOVER_BLOCKED: unsafe target or entrypoint' >&2; exit 2
fi
TOP="$(git -C "$TARGET" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$TOP" || "$(realpath -e "$TOP")" != "$(realpath -e "$TARGET")" ]]; then
  echo 'B06_RECOVER_BLOCKED: target is not an independent Git working tree' >&2; exit 2
fi
# Never replace a dirty or partial checkout containing user files.
if [[ -n "$(git -C "$TARGET" status --porcelain --untracked-files=normal)" ]]; then
  echo 'B06_RECOVER_BLOCKED: submodule has local modifications/untracked files' >&2; exit 2
fi
HEAD_NOW="$(git -C "$TARGET" rev-parse --verify 'HEAD^{commit}' 2>/dev/null || true)"
if [[ "$HEAD_NOW" == "$SHA" && -s "$TARGET/$MARKER" ]]; then
  echo "B06_RECOVER_READY already_correct sha=$SHA"; exit 0
fi
# Fetch pinned object without changing user's current checkout.
if ! git -C "$TARGET" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  echo "B06_RECOVER_FETCH pinned_sha=$SHA"
  if ! git -C "$TARGET" fetch --no-tags "$AUTHOR_URL" "$SHA"; then
    echo 'B06_RECOVER_FETCH_RETRY published-main-ref'
    git -C "$TARGET" fetch --no-tags "$AUTHOR_URL" \
      '+refs/heads/main:refs/remotes/b06-author/main'
  fi
fi
if ! git -C "$TARGET" cat-file -e "$SHA^{commit}" 2>/dev/null; then
  echo 'B06_RECOVER_BLOCKED: pinned commit unavailable after fetch' >&2; exit 2
fi
# Preserve any different, clean, valid checkout by creating an immutable local
# recovery reference. Do not delete/rewrite its existing refs or commit.
if [[ -n "$HEAD_NOW" && "$HEAD_NOW" != "$SHA" ]]; then
  BACKUP_REF="refs/b06-recovery/preexisting-$HEAD_NOW"
  if git -C "$TARGET" show-ref --verify --quiet "$BACKUP_REF"; then
    EXISTING="$(git -C "$TARGET" rev-parse --verify "$BACKUP_REF^{commit}")"
    if [[ "$EXISTING" != "$HEAD_NOW" ]]; then
      echo 'B06_RECOVER_BLOCKED: conflicting backup ref' >&2; exit 2
    fi
  else
    git -C "$TARGET" update-ref "$BACKUP_REF" "$HEAD_NOW"
  fi
  echo "B06_RECOVER_BACKUP original_sha=$HEAD_NOW"
fi
# --detach preserves remote refs and original commit, and refuses dirty worktree.
git -C "$TARGET" -c advice.detachedHead=false checkout --detach "$SHA"
if [[ "$(git -C "$TARGET" rev-parse --verify 'HEAD^{commit}')" != "$SHA" || ! -s "$TARGET/$MARKER" ]]; then
  echo 'B06_RECOVER_BLOCKED: pinned checkout lacks required entrypoint' >&2; exit 2
fi
echo "B06_RECOVER_READY pinned_sha=$SHA"
