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

from pathlib import Path

import pytest

from holoscan.cli.common.artifact_sources import ArtifactSources
from holoscan.cli.common.enum_types import Arch, PlatformConfiguration


class TestArtifactSource:
    def _init(self) -> None:
        self._artifact_source = ArtifactSources()
        current_file_path = Path(__file__).parent.resolve()
        source_file_sample = current_file_path / "./package-source.json"
        self._artifact_source.load(str(source_file_sample))

    def test_loads_from_edge(self, monkeypatch):
        artifact_source = ArtifactSources()
        artifact_source.download_manifest()

    def test_loads_invalid_file(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda x: "{}")

        source_file_sample = Path("some-bogus-file.json")
        artifact_sources = ArtifactSources()

        with pytest.raises(FileNotFoundError):
            artifact_sources.load(str(source_file_sample))

    @pytest.mark.parametrize(
        "arch,platform_config",
        [
            (Arch.amd64, PlatformConfiguration.dGPU),
            (Arch.arm64, PlatformConfiguration.dGPU),
            (Arch.arm64, PlatformConfiguration.iGPU),
        ],
    )
    def test_debian_package(self, arch, platform_config):
        self._init()
        assert self._artifact_source.debian_packages("1.0.3", arch, platform_config) is not None

    def test_base_images(self):
        self._init()
        assert self._artifact_source.base_images("1.0.3") is not None

    def test_build_images(self):
        self._init()
        assert self._artifact_source.build_images("1.0.3") is not None

    def test_health_probe(self):
        self._init()
        assert self._artifact_source.health_probe("1.0.3") is not None
