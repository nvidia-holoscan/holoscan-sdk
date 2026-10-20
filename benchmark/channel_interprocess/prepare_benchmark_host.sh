#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Capture host metadata and optionally apply benchmark-oriented tuning before
# collecting Channel domain-comparison benchmark results.
#
# Usage:
#   ./public/benchmark/channel_interprocess/prepare_benchmark_host.sh
#   APPLY=1 ./public/benchmark/channel_interprocess/prepare_benchmark_host.sh
#   APPLY=1 SET_NVP_MODEL=1 NVP_MODEL=0 ./public/benchmark/channel_interprocess/prepare_benchmark_host.sh
#
# Environment:
#   APPLY          if 1, apply available tuning steps; default 0 records metadata only
#   OUTPUT_DIR     metadata output directory; default public/build/channel-interprocess-host-prep-*
#   SET_NVP_MODEL  if 1 and nvpmodel exists, run nvpmodel -m ${NVP_MODEL}; default 0
#   NVP_MODEL      nvpmodel mode to apply when SET_NVP_MODEL=1; default 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PUBLIC_ROOT}/.." && pwd)"

APPLY="${APPLY:-0}"
SET_NVP_MODEL="${SET_NVP_MODEL:-0}"
NVP_MODEL="${NVP_MODEL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PUBLIC_ROOT}/build/channel-interprocess-host-prep-$(date +%Y%m%dT%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/prepare-benchmark-host.log"
META="${OUTPUT_DIR}/host-metadata.txt"

log() {
    printf '%s\n' "$*" | tee -a "${LOG}"
}

have() {
    command -v "$1" >/dev/null 2>&1
}

run_capture() {
    local label="$1"
    shift
    {
        printf '\n## %s\n' "${label}"
        printf '$'
        printf ' %q' "$@"
        printf '\n'
        "$@" 2>&1 || true
    } >>"${META}"
}

sudo_or_run() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    elif have sudo; then
        sudo "$@"
    else
        log "SKIP: sudo not available for privileged command: $*"
        return 1
    fi
}

write_cpu_governor() {
    local governor="$1"
    local path
    shopt -s nullglob
    for path in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        printf '%s\n' "${governor}" | sudo_or_run tee "${path}" >/dev/null || true
    done
    shopt -u nullglob
}

capture_metadata() {
    : >"${META}"
    run_capture "date" date -Is
    run_capture "git branch" git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD
    run_capture "git commit" git -C "${REPO_ROOT}" rev-parse HEAD
    run_capture "git status" git -C "${REPO_ROOT}" status --short
    run_capture "uname" uname -a
    if have lscpu; then
        run_capture "lscpu" lscpu
    fi
    if have nvidia-smi; then
        run_capture "nvidia-smi" nvidia-smi
    fi
    if have nvcc; then
        run_capture "nvcc" nvcc --version
    fi
    if have cpupower; then
        run_capture "cpupower frequency-info" cpupower frequency-info
    fi
    if have nvpmodel; then
        run_capture "nvpmodel" nvpmodel -q
    fi
    if have jetson_clocks; then
        run_capture "jetson_clocks" jetson_clocks --show
    fi
    if have tegrastats; then
        run_capture "tegrastats snapshot" timeout 2 tegrastats
    fi
    run_capture "cpu governors" bash -lc \
        'for p in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do [ -r "$p" ] && printf "%s %s\n" "$p" "$(cat "$p")"; done'
}

log "Writing host-prep output to ${OUTPUT_DIR}"
capture_metadata

if [[ "${APPLY}" != "1" ]]; then
    log "APPLY=0: metadata captured only. Re-run with APPLY=1 to apply available tuning steps."
    exit 0
fi

log "Applying available performance tuning steps..."

if have nvpmodel && [[ "${SET_NVP_MODEL}" == "1" ]]; then
    log "Setting nvpmodel mode ${NVP_MODEL}"
    sudo_or_run nvpmodel -m "${NVP_MODEL}" || true
elif have nvpmodel; then
    log "nvpmodel detected; not changing mode because SET_NVP_MODEL=0"
fi

if have jetson_clocks; then
    log "Running jetson_clocks"
    sudo_or_run jetson_clocks || true
fi

log "Setting CPU governors to performance where available"
write_cpu_governor performance

capture_metadata
log "Done. Metadata: ${META}"
