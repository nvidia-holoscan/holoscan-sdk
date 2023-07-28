#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Actually, VSCode will run this script in the local workspace folder but
# it's better to be explicit.

# Get 'localWorkspaceFolder' environment variable from the script path.
localWorkspaceFolder=$(git rev-parse --show-toplevel 2> /dev/null || dirname $(dirname $(realpath -s $0)))

# Get the holoscan sdk top directory.
TOP=$(readlink -f "${SCRIPT_DIR}/..")

# Install lint dependencies
${TOP}/run install_lint_deps
