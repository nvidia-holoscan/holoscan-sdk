#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run fastdds_benchmark publisher and subscriber, capture CUDA_BUF_LAT lines, and print latency stats
# (subscriber timestamp minus publisher timestamp per sample index and flow).
#
# Source: public/modules/holoipc/examples/fastdds_benchmark/scripts/ (CMake copies next to fastdds_benchmark binary).
#
# Usage:
#   ./run_latency_benchmark.sh -s SAMPLES -z BYTES [-a] [-e] [-b PATH] [-w N] [-v]
#
# Environment:
#   FASTDDS_BENCHMARK  default binary if -b not given (default name: try script dir, then PATH)
#   SUB_START_DELAY_SEC  non-negative integer seconds after starting subscriber before publisher (default: 2)
#   FASTDDS_BENCHMARK_TIMEOUT_SEC  if set to a positive integer and `timeout` is on PATH, wrap each
#                                   publisher/subscriber process with GNU timeout (SIGTERM, 10s kill-after)

set -euo pipefail

SAMPLES=""
MSG_SIZE=""
BINARY="${FASTDDS_BENCHMARK:-fastdds_benchmark}"
ACCEL=0
EAGER=0
SUB_START_DELAY_SEC="${SUB_START_DELAY_SEC:-2}"
WARMUP=0
VERBOSE=0

usage() {
  cat <<'EOF'
Run fastdds_benchmark publisher and subscriber, capture CUDA_BUF_LAT lines, and print latency stats
(subscriber timestamp minus publisher timestamp per sample index and flow).

Source: public/modules/holoipc/examples/fastdds_benchmark/scripts/ (CMake copies next to fastdds_benchmark binary).

Usage:
  ./run_latency_benchmark.sh -s SAMPLES -z BYTES [-a] [-e] [-b PATH] [-w N] [-v]

Environment:
  FASTDDS_BENCHMARK  default binary if -b not given (default name: try script dir, then PATH)
  SUB_START_DELAY_SEC  non-negative integer seconds to wait after starting subscriber before publisher (default: 2)
  FASTDDS_BENCHMARK_TIMEOUT_SEC  optional per-process wall-clock limit (requires GNU timeout on PATH)

Options:
EOF
  echo "  -s, --samples N          Number of samples (required)"
  echo "  -z, --message-size N     Payload size in bytes (required)"
  echo "  -a, --accel              Use AccelBuffer on both sides (must match manual runs)"
  echo "  -e, --eager              Subscriber uses acquire_pointer_eager (requires -a)"
  echo "  -b, --binary PATH        Path to fastdds_benchmark (default: beside this script, then PATH)"
  echo "  -w, --warmup N           Exclude sample indices 0..N-1 from statistics (default: 0)"
  echo "  -v, --verbose            Print per-index latency (µs)"
  echo "  -h, --help               Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--samples)
      if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value." >&2; exit 2; fi
      SAMPLES="$2"
      shift 2
      ;;
    -z|--message-size)
      if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value." >&2; exit 2; fi
      MSG_SIZE="$2"
      shift 2
      ;;
    -a|--accel)
      ACCEL=1
      shift
      ;;
    -e|--eager)
      EAGER=1
      shift
      ;;
    -b|--binary)
      if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value." >&2; exit 2; fi
      BINARY="$2"
      shift 2
      ;;
    -w|--warmup)
      if [[ $# -lt 2 ]]; then echo "Error: $1 requires a value." >&2; exit 2; fi
      WARMUP="$2"
      shift 2
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SAMPLES" || -z "$MSG_SIZE" ]]; then
  echo "Error: -s SAMPLES and -z BYTES are required." >&2
  usage >&2
  exit 2
fi

if ! [[ "$SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --samples must be a positive integer." >&2
  exit 2
fi
if ! [[ "$MSG_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: --message-size must be a positive integer (bytes)." >&2
  exit 2
fi

if [[ "$EAGER" -eq 1 && "$ACCEL" -ne 1 ]]; then
  echo "Error: --eager requires --accel (AccelBuffer path)." >&2
  exit 2
fi

if ! [[ "$WARMUP" =~ ^[0-9]+$ ]]; then
  echo "Error: --warmup must be a non-negative integer." >&2
  exit 2
fi
if [[ "$WARMUP" -ge "$SAMPLES" ]]; then
  echo "Error: --warmup ($WARMUP) must be less than --samples ($SAMPLES)." >&2
  exit 2
fi

if ! [[ "$SUB_START_DELAY_SEC" =~ ^[0-9]+$ ]]; then
  echo "Error: SUB_START_DELAY_SEC must be a non-negative integer (got: ${SUB_START_DELAY_SEC})." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_SIBLING_CANDIDATE="$SCRIPT_DIR/fastdds_benchmark"
# Bare name (no path): use executable next to this script — CMake copies the script beside fastdds_benchmark.
if [[ "$BINARY" != */* ]]; then
  _sibling="$SCRIPT_DIR/$BINARY"
  if [[ -x "$_sibling" ]]; then
    BINARY="$_sibling"
  fi
fi

if [[ ! -x "$BINARY" ]] && command -v "$BINARY" >/dev/null 2>&1; then
  BINARY="$(command -v "$BINARY")"
fi

if [[ ! -f "$BINARY" ]] || [[ ! -x "$BINARY" ]]; then
  echo "Error: fastdds_benchmark executable not found or not executable." >&2
  echo "  Resolved path or name: ${BINARY:-empty}" >&2
  echo "  Typical location after build (CMake copies this script beside the binary):" >&2
  echo "    ${_SIBLING_CANDIDATE}" >&2
  echo "Hint: build the Holoscan target first, for example:" >&2
  echo "    cmake --build <your-build-dir> --target fastdds_benchmark" >&2
  echo "  Then run this script from the build output directory, or set FASTDDS_BENCHMARK to the" >&2
  echo "  built binary, or pass -b PATH." >&2
  exit 2
fi

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/fastdds_lat.XXXXXX")"
cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

SUB_OUT="$WORKDIR/sub.stdout"
SUB_ERR="$WORKDIR/sub.stderr"
PUB_OUT="$WORKDIR/pub.stdout"
PUB_ERR="$WORKDIR/pub.stderr"

SUB_ARGS=(-s "$SAMPLES" -z "$MSG_SIZE")
PUB_ARGS=(-s "$SAMPLES" -z "$MSG_SIZE")
if [[ "$ACCEL" -eq 1 ]]; then
  SUB_ARGS+=(--accel)
  PUB_ARGS+=(--accel)
fi
if [[ "$EAGER" -eq 1 ]]; then
  SUB_ARGS+=(--eager)
fi

echo "Binary:    $BINARY"
echo "Samples:   $SAMPLES"
echo "Payload:   $MSG_SIZE bytes"
echo "Accel:     $([[ $ACCEL -eq 1 ]] && echo yes || echo no)"
echo "Eager sub: $([[ $EAGER -eq 1 ]] && echo yes || echo no)"
if [[ "$WARMUP" -gt 0 ]]; then
  echo "Warmup:    $WARMUP (primary stats use indices >= $WARMUP only)"
else
  echo "Warmup:    0 (primary stats: all indices; comparison block excludes index 0)"
fi
echo "Verbose:   $([[ $VERBOSE -eq 1 ]] && echo yes || echo no)"
echo "Work dir:  $WORKDIR"
if [[ -n "${FASTDDS_BENCHMARK_TIMEOUT_SEC:-}" ]]; then
  if [[ "${FASTDDS_BENCHMARK_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] && command -v timeout >/dev/null 2>&1; then
    echo "Timeout:   ${FASTDDS_BENCHMARK_TIMEOUT_SEC}s per process (GNU timeout)"
  elif [[ "${FASTDDS_BENCHMARK_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Timeout:   FASTDDS_BENCHMARK_TIMEOUT_SEC set but 'timeout' not found — not applied" >&2
  else
    echo "Timeout:   FASTDDS_BENCHMARK_TIMEOUT_SEC ignored (need a positive integer)" >&2
  fi
fi
echo ""

run_benchmark_child() {
  if [[ -n "${FASTDDS_BENCHMARK_TIMEOUT_SEC:-}" ]] \
      && [[ "${FASTDDS_BENCHMARK_TIMEOUT_SEC}" =~ ^[1-9][0-9]*$ ]] \
      && command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=10 "${FASTDDS_BENCHMARK_TIMEOUT_SEC}" "$@"
  else
    "$@"
  fi
}

run_benchmark_child "$BINARY" subscriber "${SUB_ARGS[@]}" >"$SUB_OUT" 2>"$SUB_ERR" &
sub_pid=$!

sleep "$SUB_START_DELAY_SEC"

run_benchmark_child "$BINARY" publisher "${PUB_ARGS[@]}" >"$PUB_OUT" 2>"$PUB_ERR" &
pub_pid=$!

sub_rc=0
pub_rc=0
wait "$sub_pid" || sub_rc=$?
wait "$pub_pid" || pub_rc=$?

if [[ "$sub_rc" -ne 0 ]]; then
  echo "Subscriber exited with code $sub_rc" >&2
  if [[ "$sub_rc" -eq 124 ]]; then
    echo "(exit 124 often means GNU timeout killed the process)" >&2
  fi
  echo "--- subscriber stderr ---" >&2
  cat "$SUB_ERR" >&2 || true
fi
if [[ "$pub_rc" -ne 0 ]]; then
  echo "Publisher exited with code $pub_rc" >&2
  if [[ "$pub_rc" -eq 124 ]]; then
    echo "(exit 124 often means GNU timeout killed the process)" >&2
  fi
  echo "--- publisher stderr ---" >&2
  cat "$PUB_ERR" >&2 || true
fi

if [[ "$sub_rc" -ne 0 || "$pub_rc" -ne 0 ]]; then
  echo "Skipping latency aggregation (publisher and/or subscriber did not exit successfully)." >&2
  exit 1
fi

export PUB_OUT SUB_OUT SAMPLES WARMUP VERBOSE
if command -v python3 >/dev/null 2>&1; then
  _LAT_PY=python3
elif command -v python >/dev/null 2>&1; then
  _LAT_PY=python
else
  echo "Error: python3 or python is required on PATH for latency aggregation." >&2
  exit 2
fi
"$_LAT_PY" - <<'PY'
import os
import re
import sys
import math
from statistics import mean, median, stdev

line_re = re.compile(
    r"CUDA_BUF_LAT kind=(\w+) flow=(\w+) index=(\d+) ts_ns=(\d+)"
)

def load_stamps(path, kind):
    # (index, flow) -> ts_ns
    out = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            k, flow, idx_s, ts_s = m.groups()
            if k != kind:
                continue
            idx_key = (int(idx_s), flow)
            if idx_key in out:
                print(
                    f"Warning: duplicate CUDA_BUF_LAT in {path!r} for kind={kind!r} "
                    f"index={idx_key}; using last timestamp.",
                    file=sys.stderr,
                )
            out[idx_key] = int(ts_s)
    return out

pub_path = os.environ["PUB_OUT"]
sub_path = os.environ["SUB_OUT"]
expect = int(os.environ["SAMPLES"])
warmup = int(os.environ.get("WARMUP", "0"))
verbose = os.environ.get("VERBOSE", "0") == "1"

pub = load_stamps(pub_path, "pub")
sub = load_stamps(sub_path, "sub")

flows = sorted({f for (_, f) in pub.keys()} | {f for (_, f) in sub.keys()})
if not flows:
    print("No CUDA_BUF_LAT lines found in publisher or subscriber stdout.", file=sys.stderr)
    sys.exit(1)

def percentile(sorted_ns, p):
    if not sorted_ns:
        return None
    n = len(sorted_ns)
    if n == 1:
        return float(sorted_ns[0])
    k = (n - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_ns[lo])
    return sorted_ns[lo] + (k - lo) * (sorted_ns[hi] - sorted_ns[lo])

def fmt_ns(ns):
    return f"{ns:>12,} ns  ({ns / 1000.0:>10.2f} µs)"

def print_stats_block(title, ns_list):
    if not ns_list:
        print(f"  [{title}] (no samples)\n")
        return
    ns_list = sorted(ns_list)
    print(f"  --- {title} (n={len(ns_list)}) ---")
    print(f"  min:    {fmt_ns(ns_list[0])}")
    print(f"  max:    {fmt_ns(ns_list[-1])}")
    print(f"  mean:   {fmt_ns(int(round(mean(ns_list))))}")
    print(f"  median: {fmt_ns(int(round(median(ns_list))))}")
    if len(ns_list) >= 2:
        print(f"  stdev:  {fmt_ns(int(round(stdev(ns_list))))}")
    for p in (90, 99):
        v = int(round(percentile(ns_list, p)))
        print(f"  p{p}:    {fmt_ns(v)}")
    print()

print("=== Latency (subscriber ts_ns − publisher ts_ns) ===\n")

any_ok = False
exit_code = 0
for flow in flows:
    pairs = []
    missing_pub = []
    missing_sub = []
    for i in range(expect):
        key = (i, flow)
        if key not in pub:
            missing_pub.append(i)
            continue
        if key not in sub:
            missing_sub.append(i)
            continue
        d = sub[key] - pub[key]
        pairs.append((i, d))

    print(f"flow={flow}")
    print(f"  expected samples: {expect}")
    print(f"  matched pairs:    {len(pairs)}")
    if missing_pub:
        print(f"  missing publisher stamp for indices: {missing_pub[:20]}{' ...' if len(missing_pub) > 20 else ''}")
    if missing_sub:
        print(f"  missing subscriber stamp for indices: {missing_sub[:20]}{' ...' if len(missing_sub) > 20 else ''}")

    neg_idx = [i for i, d in pairs if d < 0]
    if neg_idx:
        tail = f"{neg_idx[:20]}{' ...' if len(neg_idx) > 20 else ''}"
        print(
            f"  Warning: negative latency (sub − pub) for indices {tail} — "
            "check clock sync or message ordering.",
            file=sys.stderr,
        )

    if not pairs:
        print("  (no pairs to summarize)\n")
        exit_code = 1
        continue

    any_ok = True
    pairs.sort(key=lambda t: t[0])

    if verbose:
        print("  per-index latency:")
        for i, d in pairs:
            print(f"    index={i:>4}  {d / 1000.0:>10.2f} µs  ({d:,} ns)")
        print()

    all_vals = [d for _i, d in pairs]

    if warmup > 0:
        post_warmup = [d for i, d in pairs if i >= warmup]
        print_stats_block(f"indices >= {warmup} (warmup excluded: 0..{warmup - 1})", post_warmup)
        if len(post_warmup) < 2:
            exit_code = 1
    else:
        print_stats_block("all indices", all_vals)
        # Compare with first index dropped — often higher latency (discovery / caches).
        tail_vals = [d for i, d in pairs if i >= 1]
        if len(tail_vals) >= 1:
            print_stats_block("indices >= 1 (first sample excluded for comparison)", tail_vals)
        if len(all_vals) >= 2 and len(tail_vals) >= 2:
            s_all = stdev(all_vals)
            s_tail = stdev(tail_vals)
            ratio = s_tail / s_all if s_all else 0.0
            print(
                f"  stdev ratio (indices>=1 vs all): {ratio:.3f}  "
                f"(well below 1.0 ⇒ first sample likely dominates spread)\n"
            )

if not any_ok:
    sys.exit(1)
sys.exit(exit_code)
PY
py_rc=$?
exit "$py_rc"
