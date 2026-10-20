#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fetch source code for APT packages installed in the image, minus any
# packages matched by an exclusion list. Keeps only compressed archives
# (.tar.*, .dsc) to save space.
#
# Requires: deb-src entries in apt sources, dpkg-dev installed, run from
# SOURCES_PATH/apt.
#
# Environment:
#   SOURCES_PATH    default /opt/sources; output goes to SOURCES_PATH/apt/
#   EXCLUDE_LISTS   whitespace-separated paths to plain-text files listing
#                   packages to skip. Each line is either a bare 'name'
#                   (skip any version) or 'name==version' (skip only that
#                   exact version). '#' comments and blank lines ignored.
#                   Typically includes the apt-before.txt base-image
#                   snapshot plus any scanner-detected exclusion file.
set -e

SOURCES_PATH="${SOURCES_PATH:-/opt/sources}"
OUTPUT_DIR="${SOURCES_PATH}/apt"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "${TEMP_DIR}"' EXIT

INSTALLED_PACKAGES="${TEMP_DIR}/apt-installed-nv.txt"
EXCLUDED_NAMES="${TEMP_DIR}/apt-exclude-n.txt"
EXCLUDED_PACKAGES="${TEMP_DIR}/apt-exclude-nv.txt"
NEW_PACKAGES_STEP1="${TEMP_DIR}/apt-new-nv.step1"
NEW_PACKAGES="${TEMP_DIR}/apt-new-nv.final"

# Snapshot of currently installed packages as 'name==version', sorted.
dpkg-query -W -f='${Package}==${Version}\n' | sort -u > "${INSTALLED_PACKAGES}"
echo "APT packages installed in this image: $(wc -l < "${INSTALLED_PACKAGES}")"

# Collect exclude entries, split into bare-name and name==version sets.
: > "${EXCLUDED_NAMES}"
: > "${EXCLUDED_PACKAGES}"
if [ -n "${EXCLUDE_LISTS:-}" ]; then
    for list in ${EXCLUDE_LISTS}; do
        if [ ! -f "${list}" ]; then
            echo "WARNING: EXCLUDE_LISTS entry not found: ${list}" >&2
            continue
        fi
        grep -vE '^\s*(#|$)' "${list}" | awk '
            /==/ { print >> excluded_packages; next }
                 { print >> excluded_names }' \
            excluded_packages="${EXCLUDED_PACKAGES}" excluded_names="${EXCLUDED_NAMES}"
    done
    sort -u "${EXCLUDED_NAMES}" -o "${EXCLUDED_NAMES}"
    sort -u "${EXCLUDED_PACKAGES}" -o "${EXCLUDED_PACKAGES}"
fi

# Apply version-specific excludes (exact 'name==version' match).
comm -23 "${INSTALLED_PACKAGES}" "${EXCLUDED_PACKAGES}" > "${NEW_PACKAGES_STEP1}"

# Apply bare-name excludes (skip any installed version of these packages).
# EXCLUDED_BY_NAME is populated from bare 'name' entries in any exclusion file —
# in practice these come from scanner entries where version parsing failed.
# Contrast with EXCLUDED_BY_VERSION, which comes from exact 'name==version' entries
# (the apt-before.txt base-image snapshot and versioned scanner-detected packages).
awk -F'==' 'FILENAME == ARGV[1] {skip[$1]=1; next} !($1 in skip)' \
    "${EXCLUDED_NAMES}" "${NEW_PACKAGES_STEP1}" > "${NEW_PACKAGES}"
EXCLUDED_BY_NAME=$(awk -F'==' 'FILENAME == ARGV[1] {skip[$1]=1; next} $1 in skip' \
    "${EXCLUDED_NAMES}" "${NEW_PACKAGES_STEP1}" | tr '\n' ' ')
EXCLUDED_BY_VERSION=$(comm -12 "${INSTALLED_PACKAGES}" "${EXCLUDED_PACKAGES}" | tr '\n' ' ')

if [ -n "${EXCLUDED_BY_VERSION}${EXCLUDED_BY_NAME}" ]; then
    echo "APT [excluded]: ${EXCLUDED_BY_VERSION}${EXCLUDED_BY_NAME}"
fi
echo "APT packages to fetch after exclusion: $(wc -l < "${NEW_PACKAGES}")"

mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"

failed_packages=()
while IFS= read -r line; do
    [ -z "$line" ] && continue
    pkg="${line%%==*}"
    ver="${line#*==}"
    if ! apt-get source "${pkg}=${ver}" 2>/dev/null && ! apt-get source "$pkg" 2>/dev/null; then
        echo "ERROR: Failed to fetch APT source for ${pkg}==${ver}" >&2
        failed_packages+=("${pkg}==${ver}")
    fi
    # Keep only compressed archives and .dsc; remove extracted directories
    for d in */; do [ -d "$d" ] && rm -rf "$d"; done
done < "${NEW_PACKAGES}"

if (( ${#failed_packages[@]} > 0 )); then
    printf 'ERROR: Failed to fetch APT sources for: %s\n' "${failed_packages[*]}" >&2
    exit 1
fi

echo "APT source archives written to ${OUTPUT_DIR}"
