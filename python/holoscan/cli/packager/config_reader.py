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

import os
from pathlib import Path
from typing import Any

import yaml

from ..common.constants import DefaultValues, EnvironmentVariables
from ..common.exceptions import InvalidApplicationConfiguration
from .manifest_files import ApplicationManifest, PackageManifest
from .parameters import PackageBuildParameters


class ApplicationConfiguration:
    def __init__(self) -> None:
        self._config = None

    def read(self, path: Path):
        """Reads application configuration file

        Args:
            path (Path): path to the app configuration file

        Raises:
            FileNotFoundError: when file does not exists or is not a file
            InvalidApplicationConfiguration: error reading file
        """
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(
                f"Specified application configuration file cannot be found: {path}"
            )

        try:
            with open(path) as app_manifest_file:
                self._config = yaml.load(app_manifest_file, yaml.SafeLoader)
        except Exception as ex:
            raise InvalidApplicationConfiguration(
                f"Error reading application configuration file from '{path}'. Please check that "
                "the file is accessible and is a valid YAML file.",
                ex,
            )

        self._config_file_path = path
        self._validate()

    def _validate(self):
        if self._config is None:
            raise InvalidApplicationConfiguration(
                f"Error reading application configuration: {self._config_file_path}"
            )

        if "application" not in self._config:
            raise InvalidApplicationConfiguration(
                "Application ('application') configuration cannot be found in "
                f"{self._config_file_path}"
            )
        if "resources" not in self._config:
            raise InvalidApplicationConfiguration(
                "Resources ('resources') configuration cannot be found in "
                f"{self._config_file_path}"
            )
        if (
            "title" not in self._config["application"]
            or len(self._config["application"]["title"]) <= 0
        ):
            raise InvalidApplicationConfiguration(
                "Application configuration key/value ('application>title') "
                f"cannot be found or is empty in {self._config_file_path}"
            )
        self._application_object = self._config["application"]
        self._resource_object = self._config["resources"]

    def title(self) -> str:
        return self._application_object["title"]

    def pip_packages(self) -> Any:
        if "pip-packages" in self._application_object:
            return self._application_object["pip-packages"]
        return None

    def populate_app_manifest(
        self, build_parameters: PackageBuildParameters
    ) -> ApplicationManifest:
        application_manifest = ApplicationManifest()
        application_manifest.api_version = DefaultValues.API_VERSION
        application_manifest.command = build_parameters.command

        application_manifest.environment = {}
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_APPLICATION] = str(
            build_parameters.app_dir
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_INPUT_PATH] = str(
            build_parameters.input_dir
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_OUTPUT_PATH] = str(
            build_parameters.output_dir
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_WORKDIR] = str(
            build_parameters.working_dir
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_MODEL_PATH] = str(
            build_parameters.models_dir
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_CONFIG_PATH] = str(
            build_parameters.config_file_path
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_APP_MANIFEST_PATH] = str(
            build_parameters.app_manifest_path
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_PKG_MANIFEST_PATH] = str(
            build_parameters.package_manifest_path
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_DOCS_PATH] = str(
            build_parameters.docs_dir
        )
        application_manifest.environment[EnvironmentVariables.HOLOSCAN_LOGS_PATH] = str(
            build_parameters.logs_dir
        )

        application_manifest.input = {
            "path": build_parameters.input_dir,
            "formats": self._application_object["input-formats"]
            if "input-formats" in self._application_object
            else None,
        }

        application_manifest.output = {
            "path": build_parameters.output_dir,
            "formats": self._application_object["output-formats"]
            if "output-formats" in self._application_object
            else None,
        }

        application_manifest.readiness = None
        application_manifest.liveness = None
        application_manifest.timeout = build_parameters.timeout
        application_manifest.version = self._get_version(build_parameters)
        application_manifest.working_directory = str(build_parameters.working_dir)

        return application_manifest

    def populate_package_manifest(
        self, build_parameters: PackageBuildParameters
    ) -> PackageManifest:
        package_manifest = PackageManifest()
        package_manifest.api_version = DefaultValues.API_VERSION
        package_manifest.application_root = str(build_parameters.app_dir)
        package_manifest.model_root = str(build_parameters.models_dir)

        package_manifest.models = {}
        if build_parameters.models is not None:
            if len(build_parameters.models) == 1:
                package_manifest.models[
                    next(iter(build_parameters.models))
                ] = package_manifest.model_root
            else:
                for model in build_parameters.models:
                    package_manifest.models[model] = os.path.join(
                        package_manifest.model_root, model
                    )
        package_manifest.resources = self._resource_object
        package_manifest.version = self._get_version(build_parameters)

        return package_manifest

    def _get_version(self, build_parameters: PackageBuildParameters) -> str:
        if build_parameters.version is None:
            if "version" not in self._application_object:
                raise InvalidApplicationConfiguration(
                    "Application configuration key/value ('application>version') "
                    f"cannot be found or is empty in {self._config_file_path}"
                )

            return self._application_object["version"]

        return build_parameters.version
