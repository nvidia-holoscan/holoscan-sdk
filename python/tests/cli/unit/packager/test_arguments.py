"""
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pathlib
from argparse import Namespace

import pytest

from holoscan.cli.common.enum_types import ApplicationType, Platform, PlatformConfiguration, SdkType
from holoscan.cli.packager.arguments import PackagingArguments
from holoscan.cli.packager.manifest_files import ApplicationManifest, PackageManifest
from holoscan.cli.packager.parameters import DefaultValues


class TestPackagingArguments:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.input_args = Namespace()
        self.input_args.username = "holoscan"
        self.input_args.uid = 1000
        self.input_args.gid = 1000
        self.input_args.config = pathlib.Path("/path/to/config/file")
        self.input_args.timeout = 100
        self.input_args.version = "HoloscanVersionNum"
        self.input_args.docs = pathlib.Path("/path/to/docs")
        self.input_args.application = pathlib.Path("/path/to/my/app")
        self.input_args.no_cache = False
        self.input_args.output = pathlib.Path("/path/to/output")
        self.input_args.models = pathlib.Path("/path/to/models")
        self.input_args.build_cache = pathlib.Path("/path/to/build_cache")
        self.input_args.cmake_args = "-DARG1=A -DARG2=B"
        self.input_args.source = pathlib.Path("/path/to/source.json")
        self.input_args.platform = Platform.X64Workstation
        self.input_args.platform_config = PlatformConfiguration.dGPU
        self.input_args.includes = []
        self.input_args.additional_libs = [
            pathlib.Path("/path/to/lib"),
            pathlib.Path("/path/to/so"),
        ]

        self.source_load_called = False

    def _setup_mocks(self, monkeypatch):
        monkeypatch.setattr(
            "holoscan.cli.packager.models.Models.build", lambda x, y: (None, self.input_args.models)
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.platforms.Platform.configure_platforms",
            lambda a, b, c, d, e: (
                SdkType.Holoscan,
                "HoloscanVersionNum",
                "MonaiDeployVersionNum",
                [],
            ),
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.config_reader.ApplicationConfiguration.read",
            lambda x, y: (None, None),
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.config_reader.ApplicationConfiguration.title",
            lambda x: "Title",
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.config_reader.ApplicationConfiguration.pip_packages",
            lambda x: None,
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.config_reader.ApplicationConfiguration.populate_app_manifest",
            lambda x, y: ApplicationManifest(),
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.config_reader.ApplicationConfiguration.populate_package_manifest",
            lambda x, y: PackageManifest(),
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.config_reader.PackageBuildParameters._detect_application_type",
            lambda x: ApplicationType.PythonFile,
        )
        monkeypatch.setattr(
            "holoscan.cli.packager.config_reader.PackageBuildParameters._set_app_command",
            lambda x: None,
        )
        monkeypatch.setattr(
            "holoscan.cli.common.artifact_sources.ArtifactSources.download_manifest",
            lambda x: None,
        )

        def mock_artifact_load(x, y):
            self.source_load_called = True

        monkeypatch.setattr(
            "holoscan.cli.common.artifact_sources.ArtifactSources.load", mock_artifact_load
        )

    def test_input_args(self, monkeypatch):
        self._setup_mocks(monkeypatch)
        args = PackagingArguments(self.input_args, pathlib.Path("/temp"))

        assert args.build_parameters.app_dir == DefaultValues.HOLOSCAN_APP_DIR
        assert args.build_parameters.config_file_path == DefaultValues.HOLOSCAN_CONFIG_PATH
        assert args.build_parameters.docs == self.input_args.docs
        assert args.build_parameters.docs_dir == DefaultValues.HOLOSCAN_DOCS_DIR
        assert args.build_parameters.logs_dir == DefaultValues.HOLOSCAN_LOGS_DIR
        assert args.build_parameters.app_config_file_path == self.input_args.config
        assert (
            args.build_parameters.full_input_path
            == DefaultValues.WORK_DIR / DefaultValues.INPUT_DIR
        )
        assert (
            args.build_parameters.full_output_path
            == DefaultValues.WORK_DIR / DefaultValues.OUTPUT_DIR
        )
        assert args.build_parameters.input_dir == DefaultValues.INPUT_DIR
        assert args.build_parameters.models_dir == DefaultValues.MODELS_DIR
        assert args.build_parameters.output_dir == DefaultValues.OUTPUT_DIR
        assert args.build_parameters.timeout == self.input_args.timeout
        assert args.build_parameters.working_dir == DefaultValues.WORK_DIR
        assert args.build_parameters.app_manifest_path == DefaultValues.APP_MANIFEST_PATH
        assert args.build_parameters.package_manifest_path == DefaultValues.PKG_MANIFEST_PATH
        assert args.build_parameters.username == self.input_args.username
        assert args.build_parameters.uid == self.input_args.uid
        assert args.build_parameters.gid == self.input_args.gid
        assert args.build_parameters.tarball_output == self.input_args.output
        assert args.build_parameters.application_directory == self.input_args.application.parent
        assert args.build_parameters.application_type == ApplicationType.PythonFile
        assert args.build_parameters.application == self.input_args.application
        assert args.build_parameters.command is None
        assert args.build_parameters.no_cache is False
        assert args.build_parameters.pip_packages is None
        assert args.build_parameters.requirements_file_path is None
        assert args.build_parameters.holoscan_sdk_version == self.input_args.version
        assert args.build_parameters.title == "Title"
        assert args.build_parameters.version == "HoloscanVersionNum"
        assert args.build_parameters.command_filename == "app"
        assert args.build_parameters.sdk == SdkType.Holoscan
        assert args.build_parameters.additional_libs == self.input_args.additional_libs
        assert args.application_manifest is not None
        assert args.package_manifest is not None
        assert args.build_parameters.build_cache == self.input_args.build_cache
        assert args.build_parameters.cmake_args == self.input_args.cmake_args

        assert args.application_manifest.readiness is not None
        assert args.application_manifest.readiness["type"] == "command"
        assert args.application_manifest.readiness["command"] == [
            "/bin/grpc_health_probe",
            "-addr",
            ":8777",
        ]
        assert args.application_manifest.readiness["initialDelaySeconds"] == 1
        assert args.application_manifest.readiness["periodSeconds"] == 10
        assert args.application_manifest.readiness["timeoutSeconds"] == 1
        assert args.application_manifest.readiness["failureThreshold"] == 3

        assert args.application_manifest.liveness is not None
        assert args.application_manifest.liveness["type"] == "command"
        assert args.application_manifest.liveness["command"] == [
            "/bin/grpc_health_probe",
            "-addr",
            ":8777",
        ]
        assert args.application_manifest.liveness["initialDelaySeconds"] == 1
        assert args.application_manifest.liveness["periodSeconds"] == 10
        assert args.application_manifest.liveness["timeoutSeconds"] == 1
        assert args.application_manifest.liveness["failureThreshold"] == 3

        assert len(args.platforms) == 0

    def test_input_args_no_timeout(self, monkeypatch):
        self._setup_mocks(monkeypatch)
        self.input_args.timeout = DefaultValues.TIMEOUT

        args = PackagingArguments(self.input_args, pathlib.Path("/temp"))
        assert args.build_parameters.timeout == DefaultValues.TIMEOUT

    def test_input_args_no_version(self, monkeypatch):
        self._setup_mocks(monkeypatch)
        self.input_args.version = None

        args = PackagingArguments(self.input_args, pathlib.Path("/temp"))
        assert args.build_parameters.version is None

    def test_input_args_no_docs(self, monkeypatch):
        self._setup_mocks(monkeypatch)
        self.input_args.docs = None

        args = PackagingArguments(self.input_args, pathlib.Path("/temp"))
        assert args.build_parameters.docs is None

    def test_input_args_no_source(self, monkeypatch):
        self._setup_mocks(monkeypatch)
        self.input_args.source = None

        _ = PackagingArguments(self.input_args, pathlib.Path("/temp"))
        assert self.source_load_called is False

    def test_monai_app_sdk(self, monkeypatch):
        self._setup_mocks(monkeypatch)
        monkeypatch.setattr(
            "holoscan.cli.packager.platforms.Platform.configure_platforms",
            lambda a, b, c, d, e: (
                SdkType.MonaiDeploy,
                "HoloscanVersionNum",
                "MonaiDeployVersionNum",
                [],
            ),
        )
        args = PackagingArguments(self.input_args, pathlib.Path("/temp"))

        assert args.application_manifest.readiness is None
        assert args.application_manifest.liveness is None

        assert len(args.platforms) == 0
