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


class HoloscanSdkError(Exception):
    """Base class for exceptions in this module."""

    pass


class WrongApplicationPathError(HoloscanSdkError):
    """Raise when wrong application path is specified."""

    pass


class UnknownApplicationType(HoloscanSdkError):
    """Raise when wrong application path is specified."""

    pass


class InvalidSdk(HoloscanSdkError):
    """Raise when the SDK version or SDK file is not supported."""

    pass


class FailedToDetectSDKVersion(HoloscanSdkError):
    """Raise when unable to detect the SDK version."""

    pass


class InvalidApplicationConfiguration(HoloscanSdkError):
    """
    Raise when required configuration value cannot be found
    in the application configuration files."""

    pass


class IncompatiblePlatformConfiguration(HoloscanSdkError):
    """
    Raise when the platforms given by the user are incompatible."""

    pass


class RunContainerError(HoloscanSdkError):
    """
    Raise when an error is encountered while running the container image."""

    pass


class InvalidManifest(HoloscanSdkError):
    """
    Raise when the manifest is invalid."""

    pass


class ErrorReadingManifest(HoloscanSdkError):
    """
    Raise when the manifest is invalid."""


class ErrorDownloadingExternalAsset(HoloscanSdkError):
    """
    Raise when the manifest is invalid."""


class InvalidSourceFileError(HoloscanSdkError):
    """
    Raise when the provided artifact source file is invalid."""

    pass


class InvalidTagValue(HoloscanSdkError):
    """
    Raise when the Docker tag is invalid."""

    pass


class InvalidSharedMemoryValue(HoloscanSdkError):
    """
    Raise when the shared memory value is invalid."""

    pass
