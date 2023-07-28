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
from argparse import Namespace
from pathlib import Path
from typing import List

from ..common.artifact_sources import ArtifactSources
from ..common.constants import DefaultValues
from ..common.enum_types import SdkType
from .config_reader import ApplicationConfiguration
from .manifest_files import ApplicationManifest, PackageManifest
from .models import Models
from .parameters import PackageBuildParameters, PlatformParameters
from .platforms import Platform


class PackagingArguments:
    """Processes input arguments for packager"""

    @property
    def platforms(self) -> List[PlatformParameters]:
        return self._platforms

    @property
    def build_parameters(self) -> PackageBuildParameters:
        return self._build_parameters

    @property
    def application_manifest(self) -> ApplicationManifest:
        return self._application_manifest

    @property
    def package_manifest(self) -> PackageManifest:
        return self._package_manifest

    def __init__(self, args: Namespace, temp_dir: str) -> None:
        """
        Args:
            args (Namespace): Input arguments for Packager from CLI
        """
        self._logger = logging.getLogger("packager")

        self._platforms: List[PlatformParameters]
        self._build_parameters = PackageBuildParameters()
        self._artifact_sources = ArtifactSources()

        if args.source is not None:
            self._artifact_sources.load(args.source)

        self.build_parameters.username = args.username
        self.build_parameters.uid = args.uid
        self.build_parameters.gid = args.gid
        self.build_parameters.build_cache = args.build_cache
        self.build_parameters.config_file = args.config
        self.build_parameters.timeout = args.timeout if args.timeout else DefaultValues.TIMEOUT
        self.build_parameters.docs = args.docs if args.docs else None
        self.build_parameters.application = args.application
        self.build_parameters.no_cache = args.no_cache
        self.build_parameters.tarball_output = args.output
        self.build_parameters.cmake_args = args.cmake_args

        models = Models()
        platform = Platform(self._artifact_sources)
        if args.models is not None:
            self.build_parameters.models = models.build(args.models)

        self._read_application_config_file(args.config)
        self.build_parameters.version = (
            args.version if args.version else self.application_manifest.version
        )
        (
            self.build_parameters.sdk,
            self.build_parameters.sdk_version,
            self._platforms,
        ) = platform.configure_platforms(
            args, temp_dir, self.build_parameters.version, self.build_parameters.application_type
        )

        if self.build_parameters.sdk == SdkType.Holoscan:
            self.application_manifest.readiness = {
                "type": "command",
                "command": ["/bin/grpc_health_probe", "-addr", ":8777"],
                "initialDelaySeconds": 1,
                "periodSeconds": 10,
                "timeoutSeconds": 1,
                "failureThreshold": 3,
            }
            self.application_manifest.liveness = {
                "type": "command",
                "command": ["/bin/grpc_health_probe", "-addr", ":8777"],
                "initialDelaySeconds": 1,
                "periodSeconds": 10,
                "timeoutSeconds": 1,
                "failureThreshold": 3,
            }
        self.application_manifest.sdk = self.build_parameters.sdk.value
        self.application_manifest.sdk_version = self.build_parameters.sdk_version

    def _read_application_config_file(self, config_file_path: Path):
        self._logger.info(f"Reading application configuration from {config_file_path}...")
        app_config = ApplicationConfiguration()
        app_config.read(config_file_path)
        self.build_parameters.title = app_config.title()
        self.build_parameters.pip_packages = app_config.pip_packages()

        self._logger.info("Generating app.json...")
        self._application_manifest = app_config.populate_app_manifest(self.build_parameters)
        self._logger.info("Generating pkg.json...")
        self._package_manifest = app_config.populate_package_manifest(self.build_parameters)
