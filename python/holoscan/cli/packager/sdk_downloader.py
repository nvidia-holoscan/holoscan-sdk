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

import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

import requests

from ..common.artifact_sources import ArtifactSources
from ..common.enum_types import Arch
from ..common.exceptions import ErrorDownloadingExternalAsset, InvalidSdk


def download_health_probe_file(
    sdk_version: str,
    arch: Arch,
    temp_dir: str,
    logger: logging.Logger,
    artifact_sources: ArtifactSources,
) -> Path:
    """Download gRPC health probe for the specified architecture.

    Args:
        sdk_version (str): SDK version
        arch (Arch): binary architecture to download
        temp_dir (str): temporary location for storing downloaded file
        logger (logging.Logger): logger
        artifact_sources (ArtifactSources): artifact source

    Raises:
        ErrorDownloadingExternalAsset: when unable to download gRPC health probe

    Returns:
        Path: path to the downloaded file
    """
    target_dir = os.path.join(temp_dir, arch.name)
    target_file = os.path.join(target_dir, "grpc_health_probe")
    if os.path.exists(target_file) and os.path.isfile(target_file):
        return Path(target_file)

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    try:
        download_url = artifact_sources.health_prob[arch.value][sdk_version]
        logger.info(f"Downloading gRPC health probe from {download_url}...")
        response = requests.get(download_url)
        if not response.ok:
            raise ErrorDownloadingExternalAsset(
                f"failed to download health probe utility from {download_url} with "
                "HTTP status {response.status_code}."
            )
    except Exception as e:
        raise ErrorDownloadingExternalAsset(f"error downloading health probe: {e}")

    try:
        logger.info(f"Saving gRPC health probe to {target_file}...")
        with open(target_file, "wb") as f:
            f.write(response.content)
        return Path(target_file)
    except Exception as e:
        raise ErrorDownloadingExternalAsset(f"error saving health probe: {e}")


def download_sdk_debian_file(
    debian_package_source: str,
    sdk_version: str,
    arch: Arch,
    temp_dir: str,
    logger: logging.Logger,
    artifact_sources: ArtifactSources,
) -> Path:
    """Download Holoscan SDK Debian package for the specified SDK version and architecture.

    Args:
        debian_package_source(str): URI to download the Debian package from
        sdk_version (str): SDK version
        arch (Arch): Architecture
        temp_dir (str): temporary location for storing downloaded file
        logger (logging.Logger): logger
        artifact_sources (ArtifactSources): artifact source

    Raises:
        InvalidSdk: when unable to download the Holoscan SDK Debian package

    Returns:
        Path: path to the downloaded file
    """
    try:
        logger.info(
            f"Downloading Holoscan Debian package ({arch.name}) from {debian_package_source}..."
        )
        response = requests.get(debian_package_source)
        if not response.ok:
            raise InvalidSdk(
                f"failed to download SDK from {debian_package_source} with "
                "HTTP status {response.status_code}."
            )
    except Exception as ex:
        raise InvalidSdk(f"failed to download SDK from {debian_package_source}", ex)

    try:
        z = zipfile.ZipFile(BytesIO(response.content))
        unzip_dir = os.path.join(temp_dir, f"{sdk_version}_{arch.name}")
        logger.info(f"Extracting Debian Package to {unzip_dir}...")
        z.extractall(unzip_dir)
    except Exception as ex:
        raise InvalidSdk(f"failed to unzip SDK from {debian_package_source}", ex)

    for file in os.listdir(unzip_dir):
        if file.endswith(".deb"):
            file_path = os.path.join(unzip_dir, file)
            logger.info(f"Debian package for {arch.name} downloaded {file_path}")
            return Path(file_path)

    raise InvalidSdk(f"Debian package not found in {debian_package_source}")
