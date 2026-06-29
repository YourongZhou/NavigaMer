#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT="$SCRIPT_DIR/schedule_codex_resume.sh"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

output=$(
  "$SCRIPT" \
    --dry-run \
    --run-at now \
    --session-id 00000000-0000-0000-0000-000000000000 \
    --worktree "$tmpdir/worktree" \
    --out-root "$tmpdir/out" \
    --prompt "continue test"
)

grep -F "mode=dry-run" <<<"$output" >/dev/null
grep -F "session_id=00000000-0000-0000-0000-000000000000" <<<"$output" >/dev/null
grep -F "codex exec resume" <<<"$output" >/dev/null
grep -F "continue test" <<<"$output" >/dev/null

if [[ -e "$tmpdir/out/STATUS" ]]; then
  echo "dry-run should not create STATUS" >&2
  exit 1
fi
