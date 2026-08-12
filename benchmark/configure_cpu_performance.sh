#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
shopt -s nullglob

readonly CPU_SYSFS_ROOT="${HOLOSCAN_CPU_SYSFS_ROOT:-/sys/devices/system/cpu}"
readonly STATE_HEADER="# holoscan-benchmark-cpu-state-v1"
readonly IDLE_DISABLE_LATENCY_US=1

usage() {
  cat <<'EOF'
Usage:
  configure_cpu_performance.sh status
  configure_cpu_performance.sh recommend [COUNT]
  configure_cpu_performance.sh apply STATE_FILE
  configure_cpu_performance.sh restore STATE_FILE

recommend prints a comma-separated CPU list containing one online logical CPU
per physical core. It keeps the selection on the socket with the most available
cores and defaults to eight CPUs. The deterministic recommendation does not
sample transient CPU utilization or interrupt activity.

apply saves every targeted idle-state disable value and CPU-frequency governor
before changing anything. It then disables every exposed CPU idle state whose
reported exit latency is at least 1 microsecond and sets every cpufreq policy
to the performance governor. This uses the disable threshold from
`cpupower idle-set -D 1` without changing lower-latency states or depending on
architecture-specific state names.

restore writes the exact saved values back. Run apply and restore through sudo
on a normal sysfs mount:

  sudo ./benchmark/configure_cpu_performance.sh apply /tmp/holoscan-cpu.state
  sudo ./benchmark/configure_cpu_performance.sh restore /tmp/holoscan-cpu.state
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

read_value() {
  local path="$1"
  tr -d '\n' < "$path"
}

collect_idle_paths() {
  local state_dir
  local state_latency
  for state_dir in "$CPU_SYSFS_ROOT"/cpu[0-9]*/cpuidle/state*; do
    [[ -r "$state_dir/latency" && -e "$state_dir/disable" ]] || continue
    state_latency=$(read_value "$state_dir/latency")
    [[ "$state_latency" =~ ^[0-9]+$ ]] ||
      die "invalid idle-state latency in $state_dir/latency: $state_latency"
    ((state_latency >= IDLE_DISABLE_LATENCY_US)) &&
      printf '%s\n' "$state_dir/disable"
  done
}

collect_governor_paths() {
  local policies=("$CPU_SYSFS_ROOT"/cpufreq/policy*/scaling_governor)
  local path
  if ((${#policies[@]} > 0)); then
    printf '%s\n' "${policies[@]}"
    return
  fi
  for path in "$CPU_SYSFS_ROOT"/cpu[0-9]*/cpufreq/scaling_governor; do
    printf '%s\n' "$path"
  done
}

collect_core_candidates() {
  local core
  local cpu
  local cpu_dir
  local online
  local package
  local key
  local -A core_cpus=()
  local -A core_ids=()
  local -A core_packages=()

  for cpu_dir in "$CPU_SYSFS_ROOT"/cpu[0-9]*; do
    [[ -d "$cpu_dir" ]] || continue
    cpu=${cpu_dir##*/cpu}
    [[ "$cpu" =~ ^[0-9]+$ ]] || continue
    if [[ -r "$cpu_dir/online" ]]; then
      online=$(read_value "$cpu_dir/online")
      [[ "$online" == "1" ]] || continue
    fi
    package=0
    core=$cpu
    [[ -r "$cpu_dir/topology/physical_package_id" ]] &&
      package=$(read_value "$cpu_dir/topology/physical_package_id")
    [[ -r "$cpu_dir/topology/core_id" ]] &&
      core=$(read_value "$cpu_dir/topology/core_id")
    key="$package:$core"
    if [[ -z "${core_cpus[$key]:-}" ]] || ((cpu < core_cpus[$key])); then
      core_cpus[$key]=$cpu
      core_ids[$key]=$core
      core_packages[$key]=$package
    fi
  done
  for key in "${!core_cpus[@]}"; do
    printf '%s\t%s\t%s\n' \
      "${core_packages[$key]}" "${core_ids[$key]}" "${core_cpus[$key]}"
  done
}

recommend_cpus() {
  local requested="${1:-8}"
  local package
  local core
  local cpu
  local selected_package=""
  local selected_count=0
  local -a candidates
  local -a selected
  local -A package_counts=()
  local -A package_cpus=()

  [[ "$requested" =~ ^[1-9][0-9]*$ ]] ||
    die "recommend COUNT must be a positive integer"

  while IFS=$'\t' read -r package core cpu; do
    [[ -n "$package" && -n "$core" && -n "$cpu" ]] || continue
    package_counts[$package]=$(( ${package_counts[$package]:-0} + 1 ))
    package_cpus[$package]+=" $cpu"
  done < <(collect_core_candidates)

  ((${#package_counts[@]} > 0)) ||
    die "no online CPU topology found under $CPU_SYSFS_ROOT"
  for package in "${!package_counts[@]}"; do
    if ((package_counts[$package] > selected_count)) ||
      { ((package_counts[$package] == selected_count)) &&
        [[ -z "$selected_package" || "$package" -lt "$selected_package" ]]; }; then
      selected_package=$package
      selected_count=${package_counts[$package]}
    fi
  done
  ((selected_count >= requested)) ||
    die "socket $selected_package exposes only $selected_count distinct online cores"

  read -r -a candidates <<< "${package_cpus[$selected_package]}"
  mapfile -t candidates < <(printf '%s\n' "${candidates[@]}" | sort -n)
  selected=("${candidates[@]:0:requested}")
  (
    IFS=,
    printf '%s\n' "${selected[*]}"
  )
  printf 'Selected %d distinct physical cores on socket %s\n' \
    "$requested" "$selected_package" >&2
}

require_writable() {
  local path="$1"
  [[ -r "$path" ]] || die "cannot read $path"
  [[ -w "$path" ]] || die "cannot write $path; run this command through sudo"
}

performance_governor_available() {
  local governor_path="$1"
  local available_path="${governor_path%/scaling_governor}/scaling_available_governors"
  local governor
  [[ -r "$available_path" ]] || return 1
  for governor in $(< "$available_path"); do
    [[ "$governor" == "performance" ]] && return 0
  done
  return 1
}

write_value() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" > "$path"
}

validate_restore_path() {
  local kind="$1"
  local path="$2"
  local relative

  [[ "$path" == "$CPU_SYSFS_ROOT/"* ]] ||
    die "state entry is outside the CPU sysfs root: $path"
  relative=${path#"$CPU_SYSFS_ROOT/"}
  case "$kind" in
    idle)
      [[ "$relative" =~ ^cpu[0-9]+/cpuidle/state[0-9]+/disable$ ]] ||
        die "invalid idle-state restore path: $path"
      ;;
    governor)
      [[ "$relative" =~ ^cpufreq/policy[0-9]+/scaling_governor$ ||
        "$relative" =~ ^cpu[0-9]+/cpufreq/scaling_governor$ ]] ||
        die "invalid governor restore path: $path"
      ;;
  esac
}

restore_state() {
  local state_file="$1"
  local kind
  local path
  local value
  local index
  local -a restore_paths=()
  local -a restore_values=()

  [[ -r "$state_file" ]] || die "state file does not exist or is unreadable: $state_file"
  [[ "$(head -n 1 "$state_file")" == "$STATE_HEADER" ]] ||
    die "unrecognized state file: $state_file"

  while IFS=$'\t' read -r kind path value; do
    [[ "$kind" == \#* || -z "$kind" ]] && continue
    [[ "$kind" == "idle" || "$kind" == "governor" ]] ||
      die "invalid state entry kind: $kind"
    [[ -n "$value" ]] || die "empty saved value for $path"
    validate_restore_path "$kind" "$path"
    require_writable "$path"
    restore_paths+=("$path")
    restore_values+=("$value")
  done < "$state_file"

  for index in "${!restore_paths[@]}"; do
    write_value "${restore_paths[$index]}" "${restore_values[$index]}"
  done
}

apply_settings() {
  local state_file="$1"
  local state_dir
  local temporary_state
  local path
  local -a idle_paths
  local -a governor_paths

  [[ ! -e "$state_file" ]] ||
    die "state file already exists; restore or choose another path: $state_file"
  state_dir=$(dirname "$state_file")
  [[ -d "$state_dir" && -w "$state_dir" ]] ||
    die "state-file directory is not writable: $state_dir"

  mapfile -t idle_paths < <(collect_idle_paths)
  mapfile -t governor_paths < <(collect_governor_paths)
  ((${#governor_paths[@]} > 0)) ||
    die "no CPU-frequency governor controls found under $CPU_SYSFS_ROOT"

  for path in "${idle_paths[@]}" "${governor_paths[@]}"; do
    require_writable "$path"
  done
  for path in "${governor_paths[@]}"; do
    performance_governor_available "$path" ||
      die "performance governor is unavailable for $path"
  done

  temporary_state=$(mktemp "$state_dir/.holoscan-cpu-state.XXXXXX")
  {
    printf '%s\n' "$STATE_HEADER"
    for path in "${idle_paths[@]}"; do
      printf 'idle\t%s\t%s\n' "$path" "$(read_value "$path")"
    done
    for path in "${governor_paths[@]}"; do
      printf 'governor\t%s\t%s\n' "$path" "$(read_value "$path")"
    done
  } > "$temporary_state"
  mv "$temporary_state" "$state_file"

  if ! (
    for path in "${idle_paths[@]}"; do
      write_value "$path" 1
    done
    for path in "${governor_paths[@]}"; do
      write_value "$path" performance
    done
  ); then
    printf 'error: failed to apply all settings; restoring saved state\n' >&2
    restore_state "$state_file"
    return 1
  fi

  if ((${#idle_paths[@]} == 0)); then
    printf 'warning: no idle states with exit latency >= %d us were exposed under %s\n' \
      "$IDLE_DISABLE_LATENCY_US" "$CPU_SYSFS_ROOT" >&2
  fi
  printf 'Saved original CPU settings to %s\n' "$state_file"
  printf 'Disabled %d idle state(s) with exit latency >= %d us and set %d governor policy/policies to performance\n' \
    "${#idle_paths[@]}" "$IDLE_DISABLE_LATENCY_US" "${#governor_paths[@]}"
}

show_status() {
  local path
  local state_dir

  printf 'Idle states:\n'
  printf '%-8s %-8s %-12s %-10s %s\n' "CPU" "STATE" "LATENCY_US" "DISABLED" "PATH"
  for path in "$CPU_SYSFS_ROOT"/cpu[0-9]*/cpuidle/state*/disable; do
    state_dir=${path%/disable}
    printf '%-8s %-8s %-12s %-10s %s\n' \
      "$(basename "$(dirname "$(dirname "$state_dir")")")" \
      "$(read_value "$state_dir/name")" \
      "$(read_value "$state_dir/latency")" \
      "$(read_value "$path")" \
      "$path"
  done

  printf '\nCPU-frequency governors:\n'
  for path in $(collect_governor_paths); do
    printf '%-16s %s\n' "$(basename "$(dirname "$path")")" "$(read_value "$path")"
  done
}

case "${1:-}" in
  status)
    [[ $# -eq 1 ]] || die "status does not accept a state file"
    show_status
    ;;
  recommend)
    [[ $# -le 2 ]] || die "recommend accepts at most one COUNT"
    recommend_cpus "${2:-8}"
    ;;
  apply)
    [[ $# -eq 2 ]] || die "apply requires STATE_FILE"
    apply_settings "$2"
    ;;
  restore)
    [[ $# -eq 2 ]] || die "restore requires STATE_FILE"
    restore_state "$2"
    printf 'Restored CPU settings from %s\n' "$2"
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
