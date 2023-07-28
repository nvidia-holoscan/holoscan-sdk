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
import os
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple

from ..common.dockerutils import create_or_use_network, docker_run, image_exists
from ..common.exceptions import ErrorReadingManifest
from ..common.utils import get_requested_gpus, print_manifest_json, run_cmd
from .resources import get_shared_memory_size

logger = logging.getLogger("runner")


def _fetch_map_manifest(map_name: str) -> Tuple[dict, dict]:
    """
    Execute HAP/MAP and fetch the manifest files.

    Args:
        map_name: HAP/MAP image name.

    Returns:
        app_info: application manifest as a python dict
        pkg_info: package manifest as a python dict
        returncode: command return code
    """
    logger.info("Reading HAP/MAP manifest...")

    with tempfile.TemporaryDirectory() as info_dir:
        cmd = f"""docker_id=$(docker create {map_name})
docker cp $docker_id:/etc/holoscan/app.json "{info_dir}/app.json"
docker cp $docker_id:/etc/holoscan/pkg.json "{info_dir}/pkg.json"
docker rm -v $docker_id > /dev/null
"""
        returncode = run_cmd(cmd)
        if returncode != 0:
            raise ErrorReadingManifest("Error reading manifest file form the package.")

        app_json = Path(f"{info_dir}/app.json")
        pkg_json = Path(f"{info_dir}/pkg.json")

        app_info = json.loads(app_json.read_text())
        pkg_info = json.loads(pkg_json.read_text())

        print_manifest_json(app_info, "app.json")
        print_manifest_json(pkg_info, "pkg.json")

        return app_info, pkg_info


def _run_app(args: Namespace, app_info: dict, pkg_info: dict):
    """
    Executes the Holoscan Application.

    Args:
        args: user arguments
        app_info: application manifest dictionary
        pkg_info: package manifest dictionary

    Returns:
        returncode: command returncode
    """

    map_name: str = args.map
    input_path: Path = args.input
    output_path: Path = args.output
    quiet: bool = args.quiet
    driver: bool = args.driver
    worker: bool = args.worker
    fragments: str = args.fragments if args.fragments else None
    network: str = create_or_use_network(args.network, map_name)
    nic: str = args.nic if args.nic else None
    config: Path = args.config if args.config else None
    address: str = args.address if args.address else None
    worker_address: str = args.worker_address if args.worker_address else None
    render: bool = args.render
    user: str = f"{args.uid}:{args.gid}"
    hostname: str = None
    terminal: bool = args.terminal
    shared_memory_size: Optional[str] = (
        get_shared_memory_size(pkg_info, worker, driver, fragments, args.shm_size)
        if args.shm_size
        else None
    )

    commands = []

    if driver:
        commands.append("--driver")
        logger.info("Application running in Driver mode")
    if worker:
        commands.append("--worker")
        logger.info("Application running in Worker mode")
    if fragments:
        commands.append("--fragments")
        commands.append(fragments)
        logger.info(f"Configured fragments: {fragments}")
    if address:
        commands.append("--address")
        commands.append(address)
        logger.info(f"App Driver address and port: {address}")
    if worker_address:
        commands.append("--worker_address")
        commands.append(worker_address)
        logger.info(f"App Worker address and port: {worker_address}")

    if driver:
        hostname = "driver"

    docker_run(
        hostname,
        map_name,
        input_path,
        output_path,
        app_info,
        pkg_info,
        quiet,
        commands,
        network,
        nic,
        config,
        render,
        user,
        terminal,
        shared_memory_size,
    )


def _dependency_verification(map_name: str) -> bool:
    """Check if all the dependencies are installed or not.

    Args:
        map_name: HAP/MAP name

    Returns:
        True if all dependencies are satisfied, otherwise False.
    """
    logger.info("Checking dependencies...")

    # check for docker
    prog = "docker"
    logger.info('--> Verifying if "%s" is installed...\n', prog)
    if not shutil.which(prog):
        logger.error(
            '"%s" not installed, please install it from https://docs.docker.com/engine/install/.',
            prog,
        )
        return False

    buildx_paths = [
        os.path.expandvars("$HOME/.docker/cli-plugins"),
        "/usr/local/lib/docker/cli-plugins",
        "/usr/local/libexec/docker/cli-plugins",
        "/usr/lib/docker/cli-plugins",
        "/usr/libexec/docker/cli-plugins",
    ]
    prog = "docker-buildx"
    logger.info('--> Verifying if "%s" is installed...\n', prog)
    buildx_found = False
    for path in buildx_paths:
        if shutil.which(os.path.join(path, prog)):
            buildx_found = True

    if not buildx_found:
        logger.error(
            '"%s" not installed, please install it from https://docs.docker.com/engine/install/.',
            prog,
        )
        return False

    # check for map image
    logger.info('--> Verifying if "%s" is available...\n', map_name)
    if not image_exists(map_name):
        logger.error("Unable to fetch required image.")
        return False

    return True


def _pkg_specific_dependency_verification(pkg_info: dict) -> bool:
    """Checks for any package specific dependencies.

    Currently it verifies the following dependencies:
    * If gpu has been requested by the application, verify that nvidia-docker is installed.

    Args:
        pkg_info: package manifest as a python dict

    Returns:
        True if all dependencies are satisfied, otherwise False.
    """
    requested_gpus = get_requested_gpus(pkg_info)
    if requested_gpus > 0:
        # check for NVIDIA Container TOolkit
        prog = "nvidia-ctk"
        logger.info('--> Verifying if "%s" is installed...\n', prog)
        if not shutil.which(prog):
            logger.error('"%s" not installed, please install NVIDIA Container Toolkit.', prog)
            return False

    return True


def execute_run_command(args: Namespace):
    if not _dependency_verification(args.map):
        logger.error("Execution Aborted")
        sys.exit(1)

    try:
        # Fetch application manifest from MAP
        app_info, pkg_info = _fetch_map_manifest(args.map)
    except Exception as e:
        logger.error(f"Failed to fetch MAP manifest: {e}")
        sys.exit(1)

    if not _pkg_specific_dependency_verification(pkg_info):
        logger.error("Execution Aborted")
        sys.exit(1)

    try:
        # Run Holoscan Application
        _run_app(args, app_info, pkg_info)
    except Exception as ex:
        logger.debug(ex, exc_info=True)
        logger.error(f"Error executing {args.map}: {ex}")
        sys.exit(1)
