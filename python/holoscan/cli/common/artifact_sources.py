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
from typing import Any, Optional

import requests
from packaging.version import Version

from holoscan import __version__ as holoscan_version_string

from .enum_types import PlatformConfiguration, SdkType
from .exceptions import InvalidSourceFileError, ManifestDownloadError


class ArtifactSources:
    """Provides default artifact source URLs with the ability to override."""

    SectionWheelVersion = "wheel-version"
    SectionDebianVersion = "debian-version"
    SectionBaseImages = "base-images"
    SectionBuildImages = "build-images"
    SectionHealthProbe = "health-probes"
    EdgeROToken = "eyJ2ZXIiOiIyIiwidHlwIjoiSldUIiwiYWxnIjoiUlMyNTYiLCJraWQiOiJLcXV1ZVdTTlRjSkhqTFhGLTJCSnctX0lkRnY0eVhqREJyNEdWMU5Gc2NJIn0.eyJzdWIiOiJqZnJ0QDAxZHRqNnF0ZWNmcnB6MXJrNmg2cjAwd2FkXC91c2Vyc1wvc3ZjLWhvbG9zY2FuLWNsaS1wdWJsaWMtcm8iLCJzY3AiOiJtZW1iZXItb2YtZ3JvdXBzOnN2Yy1ob2xvc2Nhbi1jbGktcHVibGljLWdyb3VwIiwiYXVkIjoiamZydEAwMWR0ajZxdGVjZnJwejFyazZoNnIwMHdhZCIsImlzcyI6ImpmcnRAMDFkdGo2cXRlY2ZycHoxcms2aDZyMDB3YWRcL3VzZXJzXC9ycGFsYW5pc3dhbXkiLCJpYXQiOjE3MDY1NzA1NjUsImp0aSI6IjlmNmEyMmM1LTk5ZTItNGRlMi1hMDhiLTQxZjg2NzIyYmJjNyJ9.Y0gfyW2F0kxiKnMhGzNCyRRE2DNrDW6CUj5ozrQiIvAbSbhohskFcFmP836PU4p3ZQTzbYk9-bBwrqoPDUaZf8p9AW9GZ3mvlU2BxK0EQ-F4oKxA1_Z7agZ0KKcmcrfWnE4Ffy53qAD8PTk5vdcznpYOBpJtF4i16j2QcXvhVGGEqUyGa7_sONdK0sevb3ZztiEoupi4gD2wPTRn30rjpGIiFSDKiswAQwoyF_SqMCQWOBEeXMISp8hkEggUpvPrESv2lbpjgaKuEJ1CikbivYTJCcoqpgH7E72FXr1sB9jfwrFD8pkjtRpGGDxN43waXy4f3Ctr8_rpbmCvwSa9iw"  # noqa: E501
    HoloscanVersion = None
    ManifestFileUrl = None

    def __init__(self) -> None:
        ArtifactSources.HoloscanVersion = ".".join(
            str(i) for i in Version(holoscan_version_string).release[0:3]
        )
        ArtifactSources.ManifestFileUrl = f"https://edge.urm.nvidia.com/artifactory/sw-holoscan-cli-generic/{ArtifactSources.HoloscanVersion}/artifacts.json"
        self._logger = logging.getLogger("common")
        self._supported_holoscan_versions = ["2.6.0"]

    @property
    def holoscan_versions(self) -> list[str]:
        # logic to dynamically fetch the supported versions
        return self._supported_holoscan_versions  # for now, return the hardcoded value

    def base_image(self, version) -> str:
        return self._data[version][SdkType.Holoscan.value][ArtifactSources.SectionBaseImages]

    def build_images(self, version) -> dict[Any, str]:
        return self._data[version][SdkType.Holoscan.value][ArtifactSources.SectionBuildImages]

    def health_probe(self, version) -> dict[Any, str]:
        return self._data[version][ArtifactSources.SectionHealthProbe]

    def load(self, uri: str):
        """Overrides the default values from a given JSON file.
           Validates top-level attributes to ensure the file is valid

        Args:
            file (Path): Path to JSON file
        """
        if uri.startswith("https"):
            self._download_manifest_internal(uri)
        elif uri.startswith("http"):
            raise ManifestDownloadError(
                "Downloading manifest files from non-HTTPS servers is not supported."
            )
        else:
            self._logger.info(f"Using CLI manifest file from {uri}...")
            with open(uri) as file:
                temp = json.load(file)

            try:
                self.validate(temp)
                self._data = temp
            except Exception as ex:
                raise InvalidSourceFileError(f"{uri} is missing required data: {ex}") from ex

    def validate(self, data: Any):
        self._logger.debug("Validating CLI manifest file...")

        for key in data:
            item = data[key]
            assert SdkType.Holoscan.value in item
            holoscan = item[SdkType.Holoscan.value]

            assert ArtifactSources.SectionWheelVersion in holoscan
            assert ArtifactSources.SectionDebianVersion in holoscan
            assert ArtifactSources.SectionBaseImages in holoscan
            assert ArtifactSources.SectionBuildImages in holoscan

            for config in PlatformConfiguration:
                assert config.value in holoscan[ArtifactSources.SectionBuildImages]

    def download_manifest(self):
        self._download_manifest_internal(
            ArtifactSources.ManifestFileUrl,
            {"Authorization": f"Bearer {ArtifactSources.EdgeROToken}"},
        )

    def _download_manifest_internal(self, url, headers=None):
        self._logger.info("Downloading CLI manifest file...")
        manifest = requests.get(url, headers=headers)

        try:
            manifest.raise_for_status()
        except Exception as ex:
            raise ManifestDownloadError(
                f"Error downloading manifest file from {url}: {manifest.reason}"
            ) from ex
        else:
            self._data = manifest.json()
            self.validate(self._data)

    def debian_package_version(self, version: str) -> Optional[str]:
        """Gets the version of the Debian package based on the version of Holoscan.

        Args:
            version (str): version of Holoscan

        Returns:
            Optional[str]: Debian package version
        """
        return (
            self._data[version][SdkType.Holoscan.value][ArtifactSources.SectionDebianVersion]
            if version in self._data
            else None
        )

    def wheel_package_version(self, version: str) -> Optional[str]:
        """Gets the version of the PyPI package based on the version of Holoscan.

        Args:
            version (str): version of Holoscan

        Returns:
            Optional[str]: PyPI package version
        """
        return self._data[version][SdkType.Holoscan.value][ArtifactSources.SectionWheelVersion]
