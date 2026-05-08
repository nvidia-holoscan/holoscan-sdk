#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

APP="${1:?Usage: $0 <path/to/native_buffer_ipc_app> [count]}"
COUNT="${2:-5}"
TIMEOUT_SEC="${3:-30}"
PUB_LOG="$(mktemp)"
NATIVE_SUB_LOG="$(mktemp)"
BYTE_SUB_LOG="$(mktemp)"

cleanup() {
  rm -f "$PUB_LOG" "$NATIVE_SUB_LOG" "$BYTE_SUB_LOG"
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

echo "=== Starting native subscriber (count=${COUNT}) ==="
"$APP" --role subscriber --count "$COUNT" --native_buffer_policy preferred --label native \
  >"$NATIVE_SUB_LOG" 2>&1 &
NATIVE_SUB_PID=$!

echo "=== Starting byte subscriber (count=${COUNT}) ==="
"$APP" --role subscriber --count "$COUNT" --native_buffer_policy disabled --label byte \
  >"$BYTE_SUB_LOG" 2>&1 &
BYTE_SUB_PID=$!

sleep 2

echo "=== Starting publisher (count=${COUNT}) ==="
"$APP" --role publisher --count "$COUNT" --native_buffer_policy preferred >"$PUB_LOG" 2>&1 &
PUB_PID=$!

PUB_RC=0
wait_pid_timeout "$PUB_PID" "$TIMEOUT_SEC" "publisher" || PUB_RC=$?

NATIVE_SUB_RC=0
wait_pid_timeout "$NATIVE_SUB_PID" "$TIMEOUT_SEC" "native subscriber" || NATIVE_SUB_RC=$?

BYTE_SUB_RC=0
wait_pid_timeout "$BYTE_SUB_PID" "$TIMEOUT_SEC" "byte subscriber" || BYTE_SUB_RC=$?

echo "=== Publisher exit code: $PUB_RC ==="
echo "=== Native subscriber exit code: $NATIVE_SUB_RC ==="
echo "=== Byte subscriber exit code: $BYTE_SUB_RC ==="
echo "=== Native subscriber log ==="
sed 's/^/[native-subscriber] /' "$NATIVE_SUB_LOG"
echo "=== Byte subscriber log ==="
sed 's/^/[byte-subscriber] /' "$BYTE_SUB_LOG"
echo "=== Publisher log ==="
sed 's/^/[publisher] /' "$PUB_LOG"

if (( PUB_RC != 0 )); then
  echo "FAIL: publisher exited with code $PUB_RC"
  exit 1
fi

if (( NATIVE_SUB_RC != 0 )); then
  echo "FAIL: native subscriber exited with code $NATIVE_SUB_RC"
  exit 1
fi

if (( BYTE_SUB_RC != 0 )); then
  echo "FAIL: byte subscriber exited with code $BYTE_SUB_RC"
  exit 1
fi

if rg -q "pending exports still outstanding" "$PUB_LOG"; then
  echo "FAIL: publisher reported stale pending exports after mixed native/byte release"
  exit 1
fi

exit 0
