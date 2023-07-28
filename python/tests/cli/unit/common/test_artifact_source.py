# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest

from holoscan.cli.common.artifact_sources import ArtifactSources
from holoscan.cli.common.enum_types import Arch, PlatformConfiguration
from holoscan.cli.common.exceptions import InvalidSourceFileError


class TestArtifactSource:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._artifact_source = ArtifactSources()

    def test_default_top_level_attributes(self):
        _ = ArtifactSources()

    def test_loads_sample(self):
        current_file_path = Path(__file__).parent.resolve()
        source_file_sample = current_file_path / "../../../../holoscan/cli/package-source.json"
        artifact_sources = ArtifactSources()
        artifact_sources.load(source_file_sample)

    def test_loads_invalid_file(self, monkeypatch):
        monkeypatch.setattr(Path, "read_text", lambda x: "{}")

        source_file_sample = Path("some-bogus-file.json")
        artifact_sources = ArtifactSources()

        with pytest.raises(InvalidSourceFileError):
            artifact_sources.load(source_file_sample)

    def test_debian_package_zero_six_amd64_dgpu(self):
        assert (
            self._artifact_source.debian_packages("0.6.0", Arch.amd64, PlatformConfiguration.dGPU)
            is not None
        )
