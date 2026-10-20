#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export LC_ALL=C

usage() {
    echo "Usage: $0 <shared-library> <baseline-file>" >&2
}

if [ "$#" -ne 2 ]; then
    usage
    exit 2
fi

library_path="$1"
baseline_path="$2"

if [ ! -f "${library_path}" ]; then
    echo "Shared library not found: ${library_path}" >&2
    exit 1
fi

if [ ! -f "${baseline_path}" ]; then
    echo "Baseline file not found: ${baseline_path}" >&2
    exit 1
fi

current_symbols="$(mktemp)"
trap 'rm -f "${current_symbols}"' EXIT

nm -D --defined-only --extern-only "${library_path}" \
    | awk '{print $3}' \
    | c++filt \
    | sort -u > "${current_symbols}"

diff -u "${baseline_path}" "${current_symbols}"
