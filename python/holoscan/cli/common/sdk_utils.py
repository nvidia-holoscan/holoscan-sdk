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

import importlib.metadata
import logging
import sys
from pathlib import Path
from typing import Optional

from packaging.version import Version

from .artifact_sources import ArtifactSources
from .enum_types import SdkType
from .exceptions import FailedToDetectSDKVersion, InvalidSdk

logger = logging.getLogger("common")


def detect_sdk(sdk: Optional[SdkType] = None) -> SdkType:
    """
    Use user specified SDK or detects the SDK to use based on the executing command name.

    Args:
        sdk (Optional[SdkType]): User specified SDK.

    Returns:
        SDK (SdkType): SDK for building the application

    Raises:
        InvalidSdk: when failed to detect SDK version.
    """

    if sdk is not None:
        if not isinstance(sdk, SdkType):
            raise ValueError("sdk must be of type SdkType")
        return sdk

    command = None
    try:
        command = Path(sys.argv[0]).name.lower()
        return SdkType(command)
    except Exception:
        raise InvalidSdk(f"Invalid SDK value provided: {command}")


def detect_sdk_version(
    sdk: SdkType, artifact_sources: ArtifactSources, sdk_version: Optional[Version] = None
) -> str:
    """
    Detects SDK version to use based on installed PyPI package or user input.

    Args:
        sdk (SdkType): SDK Type.
        sdk_version (Optional[str]): SDK version to be used for building the package.

    Returns:
        Version (str): Version of the SDK to use

    Raises:
        InvalidSdk: when failed to detect SDK version.
    """
    if sdk is SdkType.Holoscan:
        return detect_holoscan_version(artifact_sources, sdk_version)
    else:
        return detect_monaideploy_version(artifact_sources, sdk_version)


def detect_holoscan_version(
    artifact_sources: ArtifactSources, sdk_version: Optional[Version] = None
) -> str:
    """
    Validates Holoscan version if specified. Otherwise, attempt to detect the Holoscan PyPI
    package installed.

    Args:
        sdk (SdkType): SDK for building the application
        sdk_version (Optional[str], optional): SDK version from user input. Defaults to None.

    Raises:
        InvalidSdk: If specified SDK version is not supported.
        FailedToDetectSDKVersion: When unable to detect the installed Holoscan PyPI package.

    Returns:
        str: SDK version
    """

    if sdk_version is not None:
        if sdk_version.base_version not in artifact_sources.holoscan_versions:
            raise InvalidSdk(
                "Invalid SDK version specified: valid values are: "
                f"{', '.join(artifact_sources.holoscan_versions)}"
            )

        return sdk_version.base_version
    else:
        try:
            ver_str = importlib.metadata.version("holoscan").title()
            ver = Version(ver_str)

            if ver.base_version not in artifact_sources.holoscan_versions:
                raise InvalidSdk(
                    f"Something must be wrong as we've detect Holoscan SDK v{ver.base_version}"
                    f" which is not supported. Please reinstall Holoscan SDK."
                )
            return ver.base_version
        except Exception as ex:
            raise FailedToDetectSDKVersion("Failed to detect installed Holoscan PyPI version.", ex)


def detect_monaideploy_version(
    artifact_sources: ArtifactSources, sdk_version: Optional[Version] = None
) -> str:
    """
    Validates MONAI Deploy version if specified. Otherwise, attempt to detect the MONAI Deploy
    PyPI package installed.

    Args:
        sdk (SdkType): SDK for building the application
        sdk_version (Optional[str], optional): SDK version from user input. Defaults to None.

    Raises:
        InvalidSdk: If specified SDK version is not supported.
        FailedToDetectSDKVersion: When unable to detect the installed MONAI Deploy PyPI package.

    Returns:
        str: SDK version
    """

    if sdk_version is not None:
        if sdk_version.base_version not in artifact_sources.monai_deploy_versions:
            raise InvalidSdk(
                "Invalid SDK version specified: valid values are: "
                f"{', '.join(artifact_sources.monai_deploy_versions)}"
            )

        return sdk_version.base_version
    else:
        try:
            ver_str = importlib.metadata.version("monai-deploy-app-sdk").title()
            ver = Version(ver_str)

            if ver.base_version not in artifact_sources.monai_deploy_versions:
                raise InvalidSdk(
                    f"Something must be wrong as we've detect MONAI App SDK v{ver.base_version}"
                    f" which is not supported. Please reinstall MONAI Deploy App SDK."
                )
            return ver.base_version
        except Exception as ex:
            raise FailedToDetectSDKVersion(
                "Failed to detect installed MONAI Deploy App SDK PyPI version.", ex
            )
