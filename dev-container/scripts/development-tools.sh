#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly INVENTORY_COMMANDS=(
    cc
    c++
    make
    cmake
    ctest
    cpack
    ninja
    pkg-config
    patchelf
    python3
    git
    curl
    rsync
    unzip
    gpg
    gdb
    gdbserver
    valgrind
    nsys
    nvcc
    cuda-gdb
)

usage() {
    echo "Usage: ${0##*/} {check|report}" >&2
}

check_capabilities() {
    local command_name
    local cmake_version
    local venv_path

    for command_name in c++ cmake ninja python3; do
        if ! command -v "${command_name}" >/dev/null 2>&1; then
            echo "Missing required development capability: ${command_name}" >&2
            return 1
        fi
    done

    cmake_version="$(cmake --version)"
    if [[ ! "${cmake_version}" =~ ^cmake\ version\ 4\. ]]; then
        printf 'CMake 4.x is required; found: %s\n' "${cmake_version%%$'\n'*}" >&2
        return 1
    fi

    python3 -m pip --version >/dev/null
    python3 -m build --version >/dev/null

    venv_path="$(mktemp -d)"
    trap 'rm -rf "${venv_path}"' RETURN
    python3 -m venv "${venv_path}"
    "${venv_path}/bin/python" --version >/dev/null

    echo "Holoscan development capabilities: PASS"
}

report_inventory() {
    local command_name
    local command_path

    printf '%-16s %s\n' "COMMAND" "STATUS"
    for command_name in "${INVENTORY_COMMANDS[@]}"; do
        if command_path="$(command -v "${command_name}" 2>/dev/null)"; then
            printf '%-16s available (%s)\n' "${command_name}" "${command_path}"
        else
            printf '%-16s missing\n' "${command_name}"
        fi
    done
}

if (( $# != 1 )); then
    usage
    exit 2
fi

case "$1" in
    check) check_capabilities ;;
    report) report_inventory ;;
    *)
        usage
        exit 2
        ;;
esac
