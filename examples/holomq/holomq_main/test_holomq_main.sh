#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <holomq_main>\n' "$0" >&2
  exit 2
fi

binary=$1
log_dir=$(mktemp -d "${TMPDIR:-/tmp}/holomq-main-ctest.XXXXXX")
publisher_log="${log_dir}/publisher.log"
subscriber_log="${log_dir}/subscriber.log"
readonly log_dir publisher_log subscriber_log
subscriber_pid=

cleanup() {
  status=$?
  trap - EXIT INT TERM

  if [[ -n "${subscriber_pid}" ]] && kill -0 "${subscriber_pid}" 2>/dev/null; then
    kill "${subscriber_pid}" 2>/dev/null || true
    wait "${subscriber_pid}" 2>/dev/null || true
  fi

  if [[ ${status} -ne 0 ]]; then
    printf '%s\n' '--- publisher ---' >&2
    cat "${publisher_log}" >&2 2>/dev/null || true
    printf '%s\n' '--- subscriber ---' >&2
    cat "${subscriber_log}" >&2 2>/dev/null || true
  fi

  # Remove only files created by this test. Leave the directory in place if it
  # contains anything unexpected rather than recursively deleting it.
  rm -f -- "${publisher_log}" "${subscriber_log}" || true
  rmdir -- "${log_dir}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 1' INT TERM

timeout 40s "${binary}" --role subscriber --samples 3 >"${subscriber_log}" 2>&1 &
subscriber_pid=$!

sleep 1
if ! timeout 20s "${binary}" --role publisher --samples 3 >"${publisher_log}" 2>&1; then
  printf 'publisher failed\n' >&2
  exit 1
fi

if ! wait "${subscriber_pid}"; then
  printf 'subscriber failed\n' >&2
  exit 1
fi
subscriber_pid=
