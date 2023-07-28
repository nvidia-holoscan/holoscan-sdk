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

import argparse
import logging
import os
from argparse import ArgumentParser, _SubParsersAction
from typing import List

from packaging.version import Version

from ..common.argparse_types import (
    valid_dir_path,
    valid_existing_dir_path,
    valid_existing_path,
    valid_platform_config,
    valid_platforms,
    valid_sdk_type,
)
from ..common.constants import SDK

logger = logging.getLogger("packager")


def create_package_parser(
    subparser: _SubParsersAction, command: str, parents: List[ArgumentParser]
) -> ArgumentParser:
    parser: ArgumentParser = subparser.add_parser(
        command, formatter_class=argparse.HelpFormatter, parents=parents, add_help=False
    )

    parser.add_argument(
        "application",
        type=valid_existing_path,
        help="Holoscan application path: Python application directory with __main__.py, "
        "Python file, C++ source directory with CMakeLists.txt, or path to an executable.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=valid_existing_path,
        help="Holoscan application configuration file (.yaml)",
    )
    parser.add_argument(
        "--docs",
        "-d",
        type=valid_existing_dir_path,
        help="path to a directory containing user documentation and/or licenses.",
    )
    parser.add_argument(
        "--models",
        "-m",
        type=valid_existing_path,
        help="path to a model file or a directory containing all models as subdirectories",
    )
    parser.add_argument(
        "--platform",
        type=valid_platforms,
        required=True,
        help="target platform(s) for the build output separated by comma. "
        f"Valid values: {str.join(', ', SDK.PLATFORMS)}.",
    )
    parser.add_argument(
        "--platform-config",
        type=valid_platform_config,
        help="target platform configuration for the build output. "
        f"Valid values: {str.join(', ', SDK.PLATFORM_CONFIGS)}.",
    )
    parser.add_argument("--timeout", type=int, help="override default application timeout")
    parser.add_argument("--version", type=Version, help="set the version of the application")

    advanced_group = parser.add_argument_group(title="advanced build options")
    advanced_group.add_argument(
        "--base-image",
        type=str,
        help="base image name for the packaged application.",
    )
    advanced_group.add_argument(
        "--build-image",
        type=str,
        help="container image name for building the C++ application.",
    )
    advanced_group.add_argument(
        "--build-cache",
        type=valid_dir_path,
        default="~/.holoscan_build_cache",
        help="path of the local directory where build cache gets stored.",
    )
    advanced_group.add_argument(
        "--cmake-args",
        type=str,
        help='additional CMAKE build arguments. E.g. "-DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_ARG=VALUE"',
    )
    advanced_group.add_argument(
        "--holoscan-sdk-file",
        type=valid_existing_path,
        help="path to the Holoscan SDK Debian or PyPI package. "
        "If not specified, the packager downloads "
        "the SDK file from the internet based on the SDK version.",
    )
    advanced_group.add_argument(
        "--monai-deploy-sdk-file",
        type=valid_existing_path,
        help="path to the MONAI Deploy App SDK PyPI package. "
        "If not specified, the packager downloads "
        "the SDK file from the internet based on the SDK version.",
    )
    advanced_group.add_argument(
        "--no-cache",
        "-n",
        dest="no_cache",
        action="store_true",
        help="do not use cache when building image",
    )
    advanced_group.add_argument(
        "--sdk",
        type=valid_sdk_type,
        help="SDK for building the application: Holoscan or MONAI-Deploy. "
        f"Valid values: {str.join(', ', SDK.SDKS)}.",
    )
    advanced_group.add_argument(
        "--source",
        type=valid_existing_path,
        help="override Debian package, build container image and run container image from a "
        "JSON formatted file.",
    )
    advanced_group.add_argument(
        "--sdk-version",
        type=Version,
        help="set the version of the SDK that is used to build and package the application. "
        "If not specified, the packager attempts to detect the installed version.",
    )

    output_group = parser.add_argument_group(title="output options")
    output_group.add_argument(
        "--output",
        "-o",
        type=valid_existing_dir_path,
        help="output directory where result images will be written.",
    )
    output_group.add_argument(
        "--tag",
        "-t",
        required=True,
        type=str,
        help="name of the image and a optional tag (format: name[:tag]).",
    )

    user_group = parser.add_argument_group(title="security options")
    user_group.add_argument(
        "--username",
        type=str,
        default="holoscan",
        help="username to be created in the container execution context.",
    )
    user_group.add_argument(
        "--uid",
        type=str,
        default=os.getuid(),
        help=f"UID associated with the username.  (default:{os.getuid()})",
    )
    user_group.add_argument(
        "--gid",
        type=str,
        default=os.getgid(),
        help=f"GID associated with the username. (default:{os.getgid()})",
    )
    parser.set_defaults(no_cache=False)
    return parser
