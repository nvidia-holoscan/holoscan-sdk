#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Add a snippet to .bashrc/.zshrc that ensures required VSCode or Cursor extensions are installed,
# so this script doesn't need to run on every container start.
# The script checks for the presence of ~/.holoscan-devcontainer-marker and installs extensions
# only if the file is missing or EXTENSIONS_ALREADY_INSTALLED is not set to "true".
rc_snippet="$(cat << 'EOF'

install_vscode_extensions() {
    local ide_name="vscode"
    local ide_bin="code"  # both 'code' and 'code-insiders' are available for VSCode Insiders
    local marker_file="$(readlink -f ~/.holoscan-devcontainer-marker)"
    local extensions_already_installed="true"

    # Load markers to see which steps have already run
    if [ -f "${marker_file}" ]; then
        source "${marker_file}"
    fi

    if [[ ${EXTENSIONS_ALREADY_INSTALLED} == "true" ]]; then
        return
    fi

    # Check if cursor command is available
    if command -v cursor &> /dev/null; then
        ide_name="cursor"
        ide_bin="cursor"
    fi

    # Install the VSCode Remote Development extension.
    # Checking if the extension is already installed is slow, so we always install it so that
    # the installation can be skipped if the extension is already installed.
    if [[ "${ide_name}" == "vscode" ]]; then
        # C/C++ Extension Pack
        "${ide_bin}" --force --install-extension ms-vscode.cpptools-extension-pack
        if [ $? -ne 0 ]; then
            extensions_already_installed="false"
        fi
        # Static type checking for Python
        "${ide_bin}" --force --install-extension ms-python.vscode-pylance
        if [ $? -ne 0 ]; then
            extensions_already_installed="false"
        fi
    elif [[ "${ide_name}" == "cursor" ]]; then
        # C/C++ Extension Pack substitute
        "${ide_bin}" --force --install-extension anysphere.cpptools
        if [ $? -ne 0 ]; then
            extensions_already_installed="false"
        fi
        # C++ Debugging support (cppdebug type in launch.json)
        "${ide_bin}" --force --install-extension KylinIdeTeam.cppdebug
        if [ $? -ne 0 ]; then
            extensions_already_installed="false"
        fi
        # Static type checking for Python
        "${ide_bin}" --force --install-extension anysphere.cursorpyright
        if [ $? -ne 0 ]; then
            extensions_already_installed="false"
        fi
    fi

    # Write marker file
    echo -e "EXTENSIONS_ALREADY_INSTALLED=${extensions_already_installed}" > ${marker_file}
}

install_vscode_extensions

EOF
)"

# Add RC snippet to ~/.bashrc and ~/.zshrc
echo "${rc_snippet}" >> ~/.bashrc
echo "${rc_snippet}" >> ~/.zshrc
