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
import tempfile
from argparse import Namespace
from typing import List

from ..common.enum_types import ApplicationType
from ..common.utils import print_manifest_json
from .arguments import PackagingArguments
from .container_builder import CppAppBuilder, PythonAppBuilder
from .manifest_files import ApplicationManifest, PackageManifest
from .parameters import PlatformBuildResults

logger = logging.getLogger("packager")


def _build_image(args: PackagingArguments, temp_dir: str) -> List[PlatformBuildResults]:
    """Creates dockerfile and builds HAP/MONAI Application Package (HAP/MAP) image
    Args:
        args (dict): Input arguments for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    if (
        args.build_parameters.application_type == ApplicationType.PythonFile
        or args.build_parameters.application_type == ApplicationType.PythonModule
    ):
        builder = PythonAppBuilder(args.build_parameters, temp_dir)
    elif (
        args.build_parameters.application_type == ApplicationType.Binary
        or args.build_parameters.application_type == ApplicationType.CppCMake
    ):
        builder = CppAppBuilder(args.build_parameters, temp_dir)

    results = []
    for platform in args.platforms:
        results.append(builder.build(platform))

    return results


def _create_app_manifest(manifest: ApplicationManifest, temp_dir: str):
    """Creates Application manifest .json file
    Args:
        manifest (Dict): Input arguments for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    map_folder_path = os.path.join(temp_dir, "map")
    os.makedirs(map_folder_path, exist_ok=True)
    with open(os.path.join(map_folder_path, "app.json"), "w") as app_manifest_file:
        app_manifest_file.write(json.dumps(manifest.data))

    print_manifest_json(manifest.data, "app.json")


def _create_package_manifest(manifest: PackageManifest, temp_dir: str):
    """Creates package manifest .json file
    Args:
        manifest (Dict): Input arguments for Packager
        temp_dir (str): Temporary directory to build MAP
    """
    map_folder_path = os.path.join(temp_dir, "map")
    os.makedirs(map_folder_path, exist_ok=True)
    with open(os.path.join(temp_dir, "map", "pkg.json"), "w") as package_manifest_file:
        package_manifest_file.write(json.dumps(manifest.data))

    print_manifest_json(manifest.data, "pkg.json")


def _package_application(args: Namespace):
    """Driver function for invoking all functions for creating and
    building the Holoscan Application package image
    Args:
        args (Namespace): Input arguments for Packager from CLI
    """
    # Initialize arguments for package
    with tempfile.TemporaryDirectory(prefix="holoscan_tmp", dir=tempfile.gettempdir()) as temp_dir:
        packaging_args = PackagingArguments(args, temp_dir)

        # Create Manifest Files
        _create_app_manifest(packaging_args.application_manifest, temp_dir)
        _create_package_manifest(packaging_args.package_manifest, temp_dir)

        results = _build_image(packaging_args, temp_dir)

        logger.info("Build Summary:")
        for result in results:
            if result.succeeded:
                print(
                    f"""\nPlatform: {result.parameters.platform.value}/{result.parameters.platform_config.value}
    Status:     Succeeded
    Docker Tag: {result.docker_tag if result.docker_tag is not None else "N/A"}
    Tarball:    {result.tarball_filenaem}"""  # noqa: E501
                )
            else:
                print(
                    f"""\nPlatform: {result.parameters.platform.value}/{result.parameters.platform_config.value}
    Status: Failure
    Error:  {result.error}
    """  # noqa: E501
                )


def execute_package_command(args: Namespace):
    try:
        _package_application(args)
    except Exception as e:
        logger.debug(e, exc_info=True)
        logger.error(f"Error packaging application:\n\n{str(e)}")
