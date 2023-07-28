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

import json
import logging
import subprocess

logger = logging.getLogger("common")


def print_manifest_json(manifest, filename):
    logger.debug(
        f"""
=============== Begin {filename} ===============
{json.dumps(manifest, indent=4)}
================ End {filename} ================
                 """
    )


def get_requested_gpus(pkg_info: dict) -> int:
    """Gets requested number of gpus in the package manifest

    Args:
        pkg_info: package manifest as a python dict

    Returns:
        int: requested number of gpus in the package manifest
    """
    num_gpu: int = pkg_info.get("resources", {}).get("gpu", 0)
    return num_gpu


def run_cmd(cmd: str) -> int:
    """
    Executes command and return the returncode of the executed command.

    Redirects stderr of the executed command to stdout.

    Args:
        cmd: command to execute.

    Returns:
        output: child process returncode after the command has been executed.
    """
    proc = subprocess.Popen(cmd, universal_newlines=True, shell=True)
    return proc.wait()
