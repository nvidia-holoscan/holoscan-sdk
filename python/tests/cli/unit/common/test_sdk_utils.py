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


import pytest
from packaging.version import Version

from holoscan.cli.common.artifact_sources import ArtifactSources
from holoscan.cli.common.enum_types import SdkType
from holoscan.cli.common.exceptions import FailedToDetectSDKVersion, InvalidSdk
from holoscan.cli.common.sdk_utils import (
    detect_holoscan_version,
    detect_monaideploy_version,
    detect_sdk,
    detect_sdk_version,
)


class TestDetectSdk:
    def test_sdk_is_not_none(self):
        assert detect_sdk(SdkType.Holoscan) == SdkType.Holoscan
        assert detect_sdk(SdkType.MonaiDeploy) == SdkType.MonaiDeploy

    def test_sdk_as_holoscan(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["/path/to/holoscan", "package"])
        assert detect_sdk() == SdkType.Holoscan

    def test_sdk_as_monai_deploy(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["/path/to/monai-deploy", "package"])
        assert detect_sdk() == SdkType.MonaiDeploy

    def test_sdk_as_unknown(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["/path/to/bla", "package"])
        with pytest.raises(InvalidSdk):
            detect_sdk()


class TestDetectSdkVersion:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._artifact_source = ArtifactSources()

    def test_sdk_version_(self, monkeypatch):
        monkeypatch.setattr(
            "holoscan.cli.common.sdk_utils.detect_holoscan_version",
            lambda x, y: SdkType.Holoscan.name,
        )
        monkeypatch.setattr(
            "holoscan.cli.common.sdk_utils.detect_monaideploy_version",
            lambda x, y: SdkType.MonaiDeploy.name,
        )
        assert detect_sdk_version(SdkType.Holoscan, self._artifact_source) == SdkType.Holoscan.name
        assert (
            detect_sdk_version(SdkType.MonaiDeploy, self._artifact_source)
            == SdkType.MonaiDeploy.name
        )


class TestDetectHoloscanVersion:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._artifact_source = ArtifactSources()

    def test_sdk_version_from_valid_user_input(self, monkeypatch):
        assert detect_holoscan_version(self._artifact_source, Version("0.6.0")) == "0.6.0"

    def test_sdk_version_from_invalid_user_input(self, monkeypatch):
        with pytest.raises(InvalidSdk):
            detect_holoscan_version(self._artifact_source, Version("0.1.0"))

    def test_detect_sdk_version(self, monkeypatch):
        version = "0.6.0"

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        result = detect_holoscan_version(self._artifact_source)
        assert result == version

    def test_detect_sdk_version_with_patch(self, monkeypatch):
        version = "0.6.0-beta-1"

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        result = detect_holoscan_version(self._artifact_source)
        assert result == "0.6.0"

    def test_detect_sdk_version_with_unsupported_version(self, monkeypatch):
        version = "0.1.2"

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        with pytest.raises(FailedToDetectSDKVersion):
            detect_holoscan_version(self._artifact_source)

    def test_detect_sdk_version_with_no_match(self, monkeypatch):
        version = "100"

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        with pytest.raises(FailedToDetectSDKVersion):
            detect_holoscan_version(self._artifact_source)


class TestDetectMonaiDeployVersion:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self._artifact_source = ArtifactSources()

    def test_sdk_version_from_valid_user_input(self, monkeypatch):
        assert detect_monaideploy_version(self._artifact_source, Version("0.6.0")) == "0.6.0"

    def test_sdk_version_from_invalid_user_input(self, monkeypatch):
        with pytest.raises(InvalidSdk):
            detect_monaideploy_version(self._artifact_source, Version("0.1.0"))

    def test_detect_sdk_version(self, monkeypatch):
        version = "0.6.0"

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        result = detect_monaideploy_version(self._artifact_source)
        assert result == version

    def test_detect_sdk_version_with_patch(self, monkeypatch):
        version = "0.6.0-beta-1"

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        result = detect_monaideploy_version(self._artifact_source)
        assert result == "0.6.0"

    def test_detect_sdk_version_with_unsupported_version(self, monkeypatch):
        version = "0.1.2"

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        with pytest.raises(FailedToDetectSDKVersion):
            detect_monaideploy_version(self._artifact_source)

    def test_detect_sdk_version_with_no_match(self, monkeypatch):
        version = Version("100")

        monkeypatch.setattr("importlib.metadata.version", lambda x: version)

        with pytest.raises(FailedToDetectSDKVersion):
            detect_monaideploy_version(self._artifact_source)
