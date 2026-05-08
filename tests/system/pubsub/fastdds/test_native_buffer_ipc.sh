#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cross-process CUDA IPC integration test launcher.
# Starts subscriber and publisher as separate OS processes, merges their
# output to stdout so that CTest / test_pattern_validation.py can check
# for pass and fail patterns.
#
# Usage:
#   test_native_buffer_ipc.sh <path/to/native_buffer_ipc_app> [count]

set -euo pipefail

APP="${1:?Usage: $0 <path/to/native_buffer_ipc_app> [count]}"
COUNT="${2:-5}"
TIMEOUT_SEC="${3:-30}"
PUB_LOG="$(mktemp)"
SUB_LOG="$(mktemp)"

cleanup() {
  rm -f "$PUB_LOG" "$SUB_LOG"
}
trap cleanup EXIT

if [[ ! -x "$APP" ]]; then
  echo "ERROR: $APP not found or not executable"
  exit 2
fi

wait_pid_timeout() {
  local pid=$1 timeout=$2 label=$3
  local elapsed=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    elapsed=$((elapsed + 1))
    if (( elapsed >= timeout )); then
      echo "TIMEOUT: $label (PID $pid) did not exit after ${timeout}s"
      kill -9 "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 124
    fi
  done
  wait "$pid"
}

echo "=== Starting subscriber (count=${COUNT}) ==="
"$APP" --role subscriber --count "$COUNT" >"$SUB_LOG" 2>&1 &
SUB_PID=$!

sleep 2

echo "=== Starting publisher (count=${COUNT}) ==="
"$APP" --role publisher --count "$COUNT" >"$PUB_LOG" 2>&1 &
PUB_PID=$!

PUB_RC=0
wait_pid_timeout "$PUB_PID" "$TIMEOUT_SEC" "publisher" || PUB_RC=$?

SUB_RC=0
wait_pid_timeout "$SUB_PID" "$TIMEOUT_SEC" "subscriber" || SUB_RC=$?

echo "=== Publisher exit code: $PUB_RC ==="
echo "=== Subscriber exit code: $SUB_RC ==="
echo "=== Subscriber log ==="
sed 's/^/[subscriber] /' "$SUB_LOG"
echo "=== Publisher log ==="
sed 's/^/[publisher] /' "$PUB_LOG"

if (( PUB_RC != 0 )); then
  echo "FAIL: publisher exited with code $PUB_RC"
  exit 1
fi

if (( SUB_RC != 0 )); then
  echo "FAIL: subscriber exited with code $SUB_RC"
  exit 1
fi

if grep -q "pending exports still outstanding" "$PUB_LOG"; then
  echo "FAIL: publisher reported stale pending exports after normal release"
  exit 1
fi

exit 0
