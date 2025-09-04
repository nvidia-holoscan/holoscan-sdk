#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script lists the dependencies of the libraries in the Holoscan SDK,
# grouped by packages, libraries, and internal dependencies.
# It is used to help with debugging issues with the Holoscan SDK.
#
# Usage:
#   ./list_ld_dependencies.sh

set -e

####################################################################
# VARS
####################################################################

site_packages=$(python3 -c 'import site; print(site.getsitepackages()[0])')

# Directories we manage without APT which we need to inspect
dirs=(
  "/opt/nvidia/holoscan/lib"
  "/workspace/holoscan-sdk/install-x86_64/lib"
  "/workspace/holoscan-sdk/install-aarch64-dgpu/lib"
  "/workspace/holoscan-sdk/install-aarch64-igpu/lib"
  "/usr/lib/ucx"
  "/opt/libtorch"
  "/opt/onnxruntime"
  "${site_packages}/cupy"
  "${site_packages}/cupy_backends/cuda"
)
dirs_pattern=$(printf "|%s" "${dirs[@]}")
dirs_pattern="${dirs_pattern:1}"  # Remove the leading '|'

# system libs to ignore
ignore_libs=(
    "libbsd.so"
    "libc.so"
    "libdl.so"
    "libgcc_s.so"
    "libm.so"
    "libmd.so"
    "libnl-3.so"
    "libpthread.so"
    "librt.so"
    "libstdc++"
    "libutil.so"
    "libz.so"
    "linux-vdso.so"
)
ignore_pattern=$(printf "|%s" "${ignore_libs[@]}")
ignore_pattern="${ignore_pattern:1}"  # Remove the leading '|'

# Arrays & maps
declare -A linkPathMap
declare -A libDependsMap
declare -A missingLibDependsMap
declare -A pkgDependsMap
declare -A internalDependsMap
declare -A missingPkgDependsMap
declare -A libReverseDependsMap
declare -A pkgReverseDependsMap
lib_ids=()
link_names=()
pkg_names=()
not_found_links=()
no_pkg_links=()

####################################################################
# UTILS
####################################################################

# Identifier for library/dir
function lib_id() {
  local lib_path=$1
  local lib_name=$(basename $lib_path)
  local dir=$(echo "$lib_path" | grep -oE "$dirs_pattern")

  # Make-up targets for holoscan libs
  if echo "$dir" | grep -q "holoscan"; then
    local target=""
    if echo "$lib_name" | grep -q "libholoscan_op_"; then
      target="holoscan::ops::${lib_name#'libholoscan_op_'}"
    elif echo "$lib_name" | grep -q "libholoscan_infer_"; then
      target="holoscan::infer::${lib_name#'libholoscan_infer_'}"
    elif echo "$lib_name" | grep -q "libholoscan_"; then
      target="holoscan::${lib_name#'libholoscan_'}"
    elif echo "$lib_name" | grep -q "libgxf_"; then
      target="holoscan::gxf::${lib_name#'libgxf_'}"
    elif echo "$lib_name" | grep -q "libuc"; then
      target="holoscan::ucx"
    elif echo "$lib_name" | grep -q "lib"; then
      target="holoscan::${lib_name#'lib'}"
    else
      target="holoscan::$lib_name"
    fi
    echo -n "${target%'.so'}"
  elif echo "$dir" | grep -q "cupy_backends/cuda"; then
    echo -n "cupy-cuda"
  else
    # And use just the dir name for our other "libs"
    echo -n $(basename $dir)
  fi
}

# Append a value to a key in an associative array
function append_to_map() {
    local -n map=$1
    local key=$2
    local value=$3
    local sep=${4:-' '}

    # Check if the key already has a value
    if [[ -z "${map[$key]}" ]]; then
        map["$key"]="$value"  # Initialize if no value exists
    else
        map["$key"]+="$sep$value"  # Append value with a space
    fi
}

# Sort values for all keys in an associative array
function sort_values_in_map() {
    local -n map=$1
    for key in "${!map[@]}"; do
      map[$key]=$(echo "${map[$key]}" | tr ' ' '\n' | sort -u)
    done
}

# Sort values in array
function sort_values_in_array() {
    local -n array=$1
    array=($(echo "${array[@]}" | tr ' ' '\n' | sort -u))
}

####################################################################
# PARSER
####################################################################

function parse_links() {
  for dir in "${dirs[@]}"; do

    if [[ ! -d $dir ]]; then
      echo "- $dir: No such file or directory"
      continue
    fi
    echo "- $dir"

    # Find all .so files and store results in an array
    mapfile -t libs < <(find "$dir" -name "*.so")

    # Iterate over each .so file
    for lib_path in "${libs[@]}"; do
      echo "  - $lib_path"

      # Identifier for library/dir
      lib_id=$(lib_id "$lib_path")
      if [[ -z "${libDependsMap[$lib_id]}" ]]; then
        lib_ids+=("$lib_id")
      fi

      # Run ldd, filter lines with '=>', then exclude lines matching any string in ignore_patterns
      # Note: we need to use grep -F to match fixed strings from our ignored patterns array, and not
      #       treat special characters (ex: ., libstdc++) as regex. We need a newline-separated list
      #       string as input for the grep -F command.
      ignore_patterns_list=$(printf '%s\n' "${ignore_libs[@]}")
      mapfile -t links < <(ldd "$lib_path" | grep "=>" | grep -vF "${ignore_patterns_list}")

      # Iterate over each line of ldd output
      for ldd_link in "${links[@]}"; do
        link_name=$(echo "$ldd_link" | awk '{print $1}')

        # Track links
        link_names+=("$link_name")
        append_to_map libDependsMap $lib_id $link_name
        append_to_map libReverseDependsMap $link_name $lib_id

        # Track missing links
        link_path=$(echo "$ldd_link" | awk '{print $3}')
        if [ $link_path = "not" ]; then
          not_found_links+=("$link_name")
          append_to_map missingLibDependsMap $lib_id $link_name
          continue
        fi

        # Track link paths
        link_path=$(realpath $link_path)
        linkPathMap["$link_name"]="$link_path"

        # Don't track dependencies between the same dir
        if echo "$link_path" | grep -q "$dir"; then
          continue
        fi

        # Track dependencies between the libraries of the dirs we're inspecting/managing as packages
        if echo "$link_path" | grep -qE "$dirs_pattern"; then
          # Add the pkg to the map to the pkg dependency map
          internal_dep=$(lib_id "$link_path")
          append_to_map internalDependsMap $lib_id $internal_dep
          continue
        fi

        # Track links with unknown packages
        pkg=$(dpkg -S $link_path 2>/dev/null | cut -d: -f1)
        if [ -z "$pkg" ]; then
          no_pkg_links+=("$link_name")
          append_to_map missingPkgDependsMap $lib_id $link_name
          continue
        fi

        # Track packages
        pkg_names+=("$pkg")
        append_to_map pkgDependsMap $lib_id $pkg
        append_to_map pkgReverseDependsMap $pkg $lib_id
      done
    done
  done

  # Sort + unique
  sort_values_in_map internalDependsMap
  sort_values_in_map pkgDependsMap
  sort_values_in_map pkgReverseDependsMap
  sort_values_in_map missingPkgDependsMap
  sort_values_in_map libDependsMap
  sort_values_in_map libReverseDependsMap
  sort_values_in_map missingLibDependsMap
  sort_values_in_array lib_ids
  sort_values_in_array link_names
  sort_values_in_array pkg_names
  sort_values_in_array no_pkg_links
  sort_values_in_array not_found_links
}

####################################################################
# PRINTERS
####################################################################

# TODO: export to json or yaml instead? then write an importer for
# displaying only the part we care about without rerunning ldd

function list_packages() {
  echo
  echo "### Needed packages ###"
  echo

  for key in "${pkg_names[@]}"; do
    echo "$key"
  done
  echo
}

function list_packages_rev_deps() {
  echo
  echo "### Needed packages ###"
  echo

  for key in "${pkg_names[@]}"; do
    echo "Package: $key"
    echo "Needed by:"
    for dep in ${pkgReverseDependsMap[$key]}; do
      echo "  - $dep"
    done
    echo
  done
}

function list_links() {
  echo
  echo "### Needed libs ###"
  echo

  for key in "${link_names[@]}"; do
    path="${linkPathMap[$key]:-'missing'}"
    echo "$key ($path)"
  done
  echo
}

function list_links_rev_deps() {
  echo
  echo "### Needed libs ###"
  echo

  for key in "${link_names[@]}"; do
    path="${linkPathMap[$key]:-'missing'}"
    echo "Library: $key ($path)"
    echo "Needed By:"
    for dep in ${libReverseDependsMap[$key]}; do
      echo "  - $dep"
    done
    echo
  done
}

function list_links_no_pkg() {
  echo
  echo "### Needed libs from unknown packages ###"
  echo

  for key in "${no_pkg_links[@]}"; do
    path="${linkPathMap[$key]:-'missing'}"
    echo "$key ($path)"
  done
  echo
}

function list_links_no_pkg_rev_deps() {
  echo
  echo "### Needed libs from unknown packages ###"
  echo

  for key in "${no_pkg_links[@]}"; do
    path="${linkPathMap[$key]:-'missing'}"
    echo "Library: $key ($path)"
    echo "Needed by:"
    for dep in ${libReverseDependsMap[$key]}; do
      echo "  - $dep"
    done
    echo
  done
}

function list_links_not_found() {
  echo
  echo "### Needed libs that are not found ###"
  echo

  for key in "${not_found_links[@]}"; do
    path="${linkPathMap[$key]:-'missing'}"
    echo "$key ($path)"
  done
  echo
}

function list_links_not_found_rev_deps() {
  echo
  echo "### Needed libs that are not found ###"
  echo

  for key in "${not_found_links[@]}"; do
    path="${linkPathMap[$key]:-'missing'}"
    echo "Library: $key ($path)"
    echo "Needed by:"
    for dep in ${libReverseDependsMap[$key]}; do
      if echo "${missingLibDependsMap[$dep]}" | grep -q $key; then
        echo "  - $dep (could not find it)"
      else
        echo "  - $dep (found it)"
      fi
    done
    echo
  done
}

function list_libs() {
  echo
  echo "### Libs ###"
  echo

  for key in "${lib_ids[@]}"; do
    echo "$key"
  done
  echo
}

function list_deps() {
  echo
  echo "### Dependency list ###"
  echo

  for key in "${lib_ids[@]}"; do
    if [[ -z "${pkgDependsMap[$key]}" ]] && [[ -z "${internalDependsMap[$key]}" ]] && [[ -z "${missingPkgDependsMap[$key]}" ]] && [[ -z "${missingLibDependsMap[$key]}" ]]; then
      continue
    fi

    echo "Library: $key"

    if [[ -n "${pkgDependsMap[$key]}" ]]; then
      echo "APT Dependencies:"
      for dep in ${pkgDependsMap[$key]}; do
        echo "  - $dep"
      done
    fi

    if [[ -n "${internalDependsMap[$key]}" ]]; then
      echo "Internal dependencies:"
      for dep in ${internalDependsMap[$key]}; do
        echo "  - $dep"
      done
    fi

    if [[ -n "${missingPkgDependsMap[$key]}" ]]; then
      echo "Other dependencies (pip, docker runtime mount, manual install...):"
      for dep in ${missingPkgDependsMap[$key]}; do
        path="${linkPathMap[$dep]:-'missing'}"
        echo "  - $dep ($path)"
      done
    fi

    if [[ -n "${missingLibDependsMap[$key]}" ]]; then
      echo "Dependencies not found (missing or no RPATH/LD_LIBRARY_PATH):"
      for dep in ${missingLibDependsMap[$key]}; do
        path="${linkPathMap[$dep]}"
        if [[ -n "$path" ]]; then
          echo "  - $dep (exists at $path)"
        else
          echo "  - $dep"
        fi
      done
    fi

    echo
  done
}

####################################################################
# DRIVERS
####################################################################

function main() {
  parse_links
  list_packages_rev_deps
  list_links_no_pkg_rev_deps
  list_links_not_found_rev_deps
  list_deps
  # list_libs
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main $@
fi
