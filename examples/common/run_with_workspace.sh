#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

# Build trees and TMPDIR can have group-writable ancestors. Use the standard
# sticky system directory, then let the example validate the complete ancestry.
# mktemp creates the private root exclusively; never change checkout permissions.
original_umask=$(umask)
umask 077
workspace_root=$(mktemp -d /tmp/holoscan-example-ctest.XXXXXXXXXX)
trap 'rm -rf -- "$workspace_root"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
umask "$original_umask"

export HOLOSCAN_EXAMPLE_WORKSPACE_ROOT="$workspace_root"
"$@"
