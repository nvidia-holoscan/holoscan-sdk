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
import os
from pathlib import Path
from typing import List

from .constants import SDK
from .enum_types import Platform, PlatformConfiguration, SdkType


def valid_dir_path(path: str, create_if_not_exists: bool = True) -> Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given path exists and it is a directory path.
    If directory does not exist, create the directory and convert string input to pathlib.Path.

    Args:
        path: string input path

    Returns:
        If path exists and is a directory, return absolute path as a pathlib.Path object.

        If path exists and is not a directory, raises argparse.ArgumentTypeError.

        If path doesn't exist, create the directory and return absolute path as a pathlib.Path object.
    """
    path = os.path.expanduser(path)
    dir_path = Path(path).absolute()
    if dir_path.exists():
        if dir_path.is_dir():
            return dir_path
        else:
            raise argparse.ArgumentTypeError(
                f"Expected directory path: '{dir_path}' is not a directory"
            )

    if create_if_not_exists:
        # create directory
        dir_path.mkdir(parents=True)
        return dir_path

    raise argparse.ArgumentTypeError(f"No such directory: '{dir_path}'")


def valid_existing_dir_path(path: str) -> Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given path exists and it is a directory path.

    Args:
        path: string input path

    Returns:
        If path exists and is a directory, return absolute path as a pathlib.Path object.

        If path doesn't exist or it is not a directory, raises argparse.ArgumentTypeError.
    """
    return valid_dir_path(path, False)


def valid_existing_path(path: str) -> Path:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert string input to pathlib.Path if the given file/folder path exists.

    Args:
        path: string input path

    Returns:
        If path exists, return absolute path as a pathlib.Path object.

        If path doesn't exist, raises argparse.ArgumentTypeError.
    """
    path = os.path.expanduser(path)
    file_path = Path(path).absolute()
    if file_path.exists():
        return file_path
    raise argparse.ArgumentTypeError(f"No such file/folder: '{file_path}'")


def valid_platforms(platforms_str: str) -> List[Platform]:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert platform strings to Platform enum if values are valid.

    Args:
        platforms_str: string comma separated platforms values
    Returns:
        If all values are valid, convert all values to Platform enum.

        Otherwise, raises argparse.ArgumentTypeError.
    """

    platforms = platforms_str.lower().split(",")
    platform_enums = []
    for platform in platforms:
        if platform not in SDK.PLATFORMS:
            raise argparse.ArgumentTypeError(f"{platform} is not a valid option for --platforms.")
        platform_enums.append(Platform(platform))

    return platform_enums


def valid_platform_config(platform_config_str: str) -> PlatformConfiguration:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert platform configuration string to PlatformConfigurations enum if value is valid.

    Args:
        platform_config_str: a platforms configuration value
    Returns:
        If the value is valid, convert the value to PlatformConfigurations enum.

        Otherwise, raises argparse.ArgumentTypeError.
    """

    platform_config_str = platform_config_str.lower()
    if platform_config_str not in SDK.PLATFORM_CONFIGS:
        raise argparse.ArgumentTypeError(
            f"{platform_config_str} is not a valid option for --platform-config."
        )

    return PlatformConfiguration(platform_config_str)


def valid_sdk_type(sdk_str: str) -> SdkType:
    """Helper type checking and type converting method for ArgumentParser.add_argument
    to convert sdk string to SdkType enum if value is valid.

    Args:
        sdk_str: sdk string
    Returns:
        If the value is valid, convert the value to SdkType enum.

        Otherwise, raises argparse.ArgumentTypeError.
    """

    sdk_str = sdk_str.lower()
    if sdk_str not in SDK.SDKS:
        raise argparse.ArgumentTypeError(f"{sdk_str} is not a valid option for --sdk.")

    return SdkType(sdk_str)
