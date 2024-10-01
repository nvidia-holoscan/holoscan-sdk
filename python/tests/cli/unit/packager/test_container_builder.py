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

import os
import pathlib
import shutil
import tempfile

import pytest

import holoscan.cli.common.dockerutils
from holoscan.cli.common.enum_types import Platform, PlatformConfiguration, SdkType
from holoscan.cli.packager.container_builder import BuilderBase
from holoscan.cli.packager.parameters import PackageBuildParameters
from holoscan.cli.packager.platforms import PlatformParameters


class TestContainerBuilder:
    @pytest.mark.parametrize(
        "no_cache",
        [(True), (False)],
    )
    def test_build_with_cache_test(self, monkeypatch, no_cache):
        """
        Tests the caching options, cache_from and cache_to based on value of the variable `cache`.
        """
        self._build_arguments = None

        def build_docker_image(**build_args):
            self._build_arguments = build_args

        monkeypatch.setattr(
            holoscan.cli.packager.container_builder,
            "create_and_get_builder",
            lambda x: "builder",
        )
        monkeypatch.setattr(
            holoscan.cli.packager.container_builder,
            "build_docker_image",
            build_docker_image,
        )
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: False)
        monkeypatch.setattr(os.path, "exists", lambda x: True)
        monkeypatch.setattr(os.path, "isfile", lambda x: True)
        monkeypatch.setattr(shutil, "rmtree", lambda path: None)
        monkeypatch.setattr(shutil, "copytree", lambda src, dest: None)
        monkeypatch.setattr(shutil, "copyfile", lambda src, dest: None)
        monkeypatch.setattr(shutil, "copy2", lambda src, dest: None)
        build_parameters = self._get_build_parameters()
        build_parameters.no_cache = no_cache
        platform_parameters = PlatformParameters(
            Platform.X64Workstation, PlatformConfiguration.dGPU, "image:tag", "1.0"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            container_builder = BuilderBase(build_parameters, temp_dir)
            container_builder.build(platform_parameters)

        assert "cache" in self._build_arguments
        assert "cache_from" in self._build_arguments
        assert "cache_to" in self._build_arguments

        assert self._build_arguments["cache"] is False if no_cache else True
        assert self._build_arguments["cache_from"] is None if no_cache else not None
        assert self._build_arguments["cache_to"] is None if no_cache else not None

    def _get_build_parameters(self):
        parameters = PackageBuildParameters()
        parameters.application = pathlib.Path("/app/app.py")
        parameters.config_file = pathlib.Path("/app/config.yaml")
        parameters.sdk = SdkType.Holoscan
        parameters.models = None
        parameters.docs = None
        parameters.tarball_output = pathlib.Path("/tarball/")
        return parameters
