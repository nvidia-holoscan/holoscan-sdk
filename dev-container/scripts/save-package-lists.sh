#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Save lists of installed APT and pip packages for later diffing.
# Run once at the start of the release stage (before any new installs).
# Used by the open-source sources stage to fetch sources only for packages
# added (or version-upgraded) by this container build.
#
# Output format: one 'name==version' entry per line, sorted. This is the
# same format consumed by fetch-apt-sources.sh (EXCLUDE_LISTS) and
# fetch-pip-sources.py (--exclude-list), so the before snapshots are just
# passed to those scripts as additional exclusion lists.
set -e

SOURCES_PATH="${SOURCES_PATH:-/opt/sources}"
mkdir -p "${SOURCES_PATH}"

dpkg-query -W -f='${Package}==${Version}\n' | sort -u > "${SOURCES_PATH}/apt-before.txt"
echo "Saved $(wc -l < "${SOURCES_PATH}/apt-before.txt") APT entries to ${SOURCES_PATH}/apt-before.txt"

python3 -m pip list --format=freeze | sort -u > "${SOURCES_PATH}/pip-before.txt"
echo "Saved $(wc -l < "${SOURCES_PATH}/pip-before.txt") pip entries to ${SOURCES_PATH}/pip-before.txt"
