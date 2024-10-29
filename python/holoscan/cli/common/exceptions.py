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


class HoloscanSdkError(Exception):
    """Base class for exceptions in this module."""

    pass


class WrongApplicationPathError(HoloscanSdkError):
    """Raise when wrong application path is specified."""

    pass


class UnknownApplicationTypeError(HoloscanSdkError):
    """Raise when wrong application path is specified."""

    pass


class InvalidSdkError(HoloscanSdkError):
    """Raise when the SDK version or SDK file is not supported."""

    pass


class FailedToDetectSDKVersionError(HoloscanSdkError):
    """Raise when unable to detect the SDK version."""

    pass


class InvalidApplicationConfigurationError(HoloscanSdkError):
    """
    Raise when required configuration value cannot be found
    in the application configuration files."""

    pass


class IncompatiblePlatformConfigurationError(HoloscanSdkError):
    """
    Raise when the platforms given by the user are incompatible."""

    pass


class RunContainerError(HoloscanSdkError):
    """
    Raise when an error is encountered while running the container image."""

    pass


class InvalidManifestError(HoloscanSdkError):
    """
    Raise when the manifest is invalid."""

    pass


class ManifestReadError(HoloscanSdkError):
    """
    Raise when the manifest is invalid."""


class ExternalAssetDownloadError(HoloscanSdkError):
    """
    Raise when the manifest is invalid."""


class InvalidSourceFileError(HoloscanSdkError):
    """
    Raise when the provided artifact source file is invalid."""

    pass


class InvalidTagValueError(HoloscanSdkError):
    """
    Raise when the Docker tag is invalid."""

    pass


class InvalidSharedMemoryValueError(HoloscanSdkError):
    """
    Raise when the shared memory value is invalid."""

    pass


class ManifestDownloadError(HoloscanSdkError):
    """
    Raise when the failed to download manifest file."""

    pass


class UnmatchedDeviceError(HoloscanSdkError):
    """
    Raise when the shared memory value is invalid."""

    def __init__(self, unmatched_devices: list[str], *args: object) -> None:
        super().__init__(
            f"The following devices cannot be found in /dev/: {str.join(',', unmatched_devices)}"
        )


class GpuResourceError(HoloscanSdkError):
    """
    Raise when the available GPUs are less than requetsed."""

    pass
