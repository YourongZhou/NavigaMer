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
assert_exact "$stdout_file" $'Usage:\n  candidate_tool --help\n  candidate_tool build --method contig --k N --ref PATH --window N --stride N --out-dir PATH\n  candidate_tool query --index PATH --reads PATH --tau N --out PATH\n  candidate_tool inspect-reference --ref PATH --window N --stride N\n  candidate_tool tensor-build --ref PATH --window N --stride N --dimension N --seed N --hnsw-m N --hnsw-ef-construction N --hnsw-ef-search N --out-dir PATH [--exact-vectors 0|1]\n  candidate_tool tensor-query --index-dir PATH --query DNA [--top-k N]'
assert_empty "$stderr_file"

reads="$test_dir/reads.fq"
index_dir="$test_dir/index"
output_tsv="$test_dir/candidates.tsv"
printf '@read1\nACGT\n+\nIIII\n' >"$reads"
run_command "$tool" build --method contig --k 3 --ref "$reference" --window 4 --stride 1 --out-dir "$index_dir"
assert_status 0
assert_empty "$stderr_file"
run_command "$tool" query --index "$index_dir/index.bin" --reads "$reads" --tau 2 --out "$output_tsv"
assert_status 0
assert_empty "$stderr_file"
assert_exact "$output_tsv" $'read_id\ttau\traw_candidate_count\tcandidate_window_ids\nread1\t2\t4\t0,1,3,4'

crlf_reads="$test_dir/reads_crlf.fq"
printf '@read1\r\nACGT\r\n+\r\nIIII\r\n' >"$crlf_reads"
run_command "$tool" query --index "$index_dir/index.bin" --reads "$crlf_reads" --tau 2 --out "$output_tsv"
assert_status 0
assert_empty "$stderr_file"
assert_exact "$output_tsv" $'read_id\ttau\traw_candidate_count\tcandidate_window_ids\nread1\t2\t4\t0,1,3,4'

multiline_reads="$test_dir/reads_multiline.fa"
printf '>read1 description\nAC\nGT\n' >"$multiline_reads"
run_command "$tool" query --index "$index_dir/index.bin" --reads "$multiline_reads" --tau 2 --out "$output_tsv"
assert_status 0
assert_empty "$stderr_file"
assert_exact "$output_tsv" $'read_id\ttau\traw_candidate_count\tcandidate_window_ids\nread1\t2\t4\t0,1,3,4'

bad_plus_reads="$test_dir/reads_bad_plus.fq"
printf '@read1\nACGT\n-\nIIII\n' >"$bad_plus_reads"
run_command "$tool" query --index "$index_dir/index.bin" --reads "$bad_plus_reads" --tau 2 --out "$output_tsv"
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: invalid FASTQ separator line in reads file'

bad_quality_reads="$test_dir/reads_bad_quality.fq"
printf '@read1\nACGT\n+\nIII\n' >"$bad_quality_reads"
run_command "$tool" query --index "$index_dir/index.bin" --reads "$bad_quality_reads" --tau 2 --out "$output_tsv"
assert_status 1
assert_empty "$stdout_file"
assert_exact "$stderr_file" 'error: FASTQ sequence and quality lengths differ in reads file'

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

rm -f "$reference"
run_command "$tool" query --index "$index_dir/index.bin" --reads "$reads" --tau 2 --out "$output_tsv"
assert_status 0
assert_empty "$stderr_file"
assert_exact "$output_tsv" $'read_id\ttau\traw_candidate_count\tcandidate_window_ids\nread1\t2\t4\t0,1,3,4'

printf 'candidate tool CLI tests passed\n'
