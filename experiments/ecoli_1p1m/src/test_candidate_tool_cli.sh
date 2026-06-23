#!/usr/bin/env bash
set -euo pipefail

tool=${1:-./candidate_tool}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/candidate_tool_cli.XXXXXX")
trap 'rm -rf "$test_dir"' EXIT

reference="$test_dir/reference.fa"
stdout_file="$test_dir/stdout"
stderr_file="$test_dir/stderr"
printf '>ecoli description\nacgt\nACgt\n' >"$reference"

run_command() {
  set +e
  "$@" >"$stdout_file" 2>"$stderr_file"
  status=$?
  set -e
}

assert_status() {
  local expected=$1
  if [[ $status -ne $expected ]]; then
    printf 'expected status %s, got %s\n' "$expected" "$status" >&2
    exit 1
  fi
}

assert_empty() {
  local path=$1
  if [[ -s $path ]]; then
    printf 'expected %s to be empty:\n' "$path" >&2
    cat "$path" >&2
    exit 1
  fi
}

assert_exact() {
  local path=$1
  local expected=$2
  local expected_file="$test_dir/expected"
  printf '%s\n' "$expected" >"$expected_file"
  if ! cmp -s "$expected_file" "$path"; then
    printf 'unexpected contents of %s:\n' "$path" >&2
    cat "$path" >&2
    exit 1
  fi
}

run_command "$tool" --help
assert_status 0
assert_exact "$stdout_file" $'Usage:\n  candidate_tool --help\n  candidate_tool inspect-reference --ref PATH --window N --stride N\n  candidate_tool tensor-build --ref PATH --window N --stride N --dimension N --seed N --hnsw-m N --hnsw-ef-construction N --hnsw-ef-search N --out-dir PATH [--exact-vectors 0|1]\n  candidate_tool tensor-query --index-dir PATH --query DNA [--top-k N]'
assert_empty "$stderr_file"

run_command "$tool" inspect-reference --ref "$reference" --window 4 --stride 1
assert_status 0
assert_exact "$stdout_file" $'contig_id\treference_length\twindow_length\tstride\tnumber_of_windows\necoli\t8\t4\t1\t5'
assert_empty "$stderr_file"

run_command "$tool" unknown-command
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: unknown command: unknown-command'

run_command "$tool" inspect-reference --ref "$reference" --window 4 --stride 1 --bogus value
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: unknown flag: --bogus'

run_command "$tool" inspect-reference --ref "$reference" --window 4 --stride
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: missing value for flag: --stride'

run_command "$tool" inspect-reference --ref "$reference" --window 4
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: missing required flag: --stride'

run_command "$tool" inspect-reference --ref "$reference" --window 4 --stride 1 --stride 2
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: duplicate flag: --stride'

run_command "$tool" inspect-reference --ref "$reference" --window nope --stride 1
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: invalid value for --window: nope'

run_command "$tool" inspect-reference --ref "$reference" --window 4294967296 --stride 1
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: invalid value for --window: 4294967296'

printf 'candidate tool CLI tests passed\n'
