"""
SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import json
import logging
import socket
import subprocess

import psutil
from packaging import version

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


def get_gpu_count():
    return len(run_cmd_output("nvidia-smi -L").splitlines())


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


def run_cmd_output(cmd: str) -> str:
    """
    Executes command and returns the output.

    Args:
        cmd: command to execute.

    Returns:
        output: command output.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return proc.stdout


def compare_versions(version1, version2):
    """
    Compares two version strings.

    Args:
        version1(str)
        version2(str)

    Returns:
        1: when version1 is greater than version2
        -1: when version1 is less than version2
        0: when two version strings are equal
    """
    v1 = version.parse(version1)
    v2 = version.parse(version2)

    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    else:
        return 0


def get_host_ip_addresses() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Returns a tuple containing interface name and its IPv4 address as the first item
    and another item with interface name and its IPv6 address.

    Returns:
        (Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]): where the item contains a list of
        tuples of network interface names and its IPv4 address. The second item is similar but
        contains IPv6 addresses.
    """
    ipv4 = []
    ipv6 = []

    for interface, snics in psutil.net_if_addrs().items():
        for snic in snics:
            if snic.family == socket.AF_INET:
                ipv4.append((interface, snic.address))
            elif snic.family == socket.AF_INET6:
                ipv6.append((interface, snic.address))

    return (ipv4, ipv6)
