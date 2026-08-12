#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <git-executable> <patch-file> [<patch-file> ...]" >&2
    exit 2
fi

git_executable="$1"
shift

patch_files=("$@")

# The iceoryx2 patch inputs are an ordered patch series. Later patches may
# intentionally edit context introduced by earlier patches, so checking every
# patch independently with `git apply --reverse --check` can falsely report an
# inconsistent tree after the full series has already been applied.
#
# The final patch is the repeatability sentinel: every patch in this series must
# depend on, or be ordered after, the preceding patch state. If a future patch is
# independent enough to be reversible without the prior patches, keep it out of
# this sentinel position or replace this helper with a temporary-tree full-series
# check.
last_patch_index=$((${#patch_files[@]} - 1))
if "${git_executable}" apply --reverse --check "${patch_files[${last_patch_index}]}" >/dev/null 2>&1; then
    exit 0
fi

for patch_file in "${patch_files[@]}"; do
    if "${git_executable}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        continue
    fi

    if "${git_executable}" apply --check "${patch_file}" >/dev/null 2>&1; then
        "${git_executable}" apply "${patch_file}"
        continue
    fi

    printf '%s\n' \
        "HoloMQ: iceoryx2 patch state is inconsistent." \
        "Patch: ${patch_file}" \
        "Git: ${git_executable}" \
        "Inspect or reset the dependency source tree, then reconfigure." >&2
    exit 1
done
