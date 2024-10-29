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

import argparse
import os
import pathlib
from pathlib import PosixPath

import pytest

from holoscan.cli.common.argparse_types import (
    valid_dir_path,
    valid_existing_dir_path,
    valid_existing_path,
    valid_platform_config,
    valid_platforms,
    valid_sdk_type,
)
from holoscan.cli.common.enum_types import Platform, PlatformConfiguration, SdkType


class TestValidDirPath:
    def test_dir_exists_and_isdir(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "is_dir", lambda x: True)
        result = valid_dir_path("/this/is/some/path")

        assert type(result) is PosixPath
        assert str(result).startswith("/this")

    def test_dir_exists_and_not_isdir(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "is_dir", lambda x: False)

        with pytest.raises(argparse.ArgumentTypeError):
            valid_dir_path("this/is/some/path")

    def test_not_dir_exists(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: False)
        monkeypatch.setattr(pathlib.Path, "mkdir", lambda x, parents: False)

        result = valid_dir_path("this/is/some/path")

        assert type(result) is PosixPath

    def test_dir_exists_and_isdir_and_expands_user_dir(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "is_dir", lambda x: True)
        result = valid_dir_path("~/this/is/some/path")

        assert type(result) is PosixPath

        assert str(result).startswith(os.path.expanduser("~"))


class TestValidExistingDirPath:
    def test_dir_path_exists_and_isdir(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "is_dir", lambda x: True)
        result = valid_existing_dir_path("this/is/some/path")

        assert type(result) is PosixPath

    @pytest.mark.parametrize("exists,isdir", [(False, False), (True, False), (False, True)])
    def test_dir_path_exists_and_isdir_combo(self, monkeypatch, exists, isdir):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: exists)
        monkeypatch.setattr(pathlib.Path, "is_dir", lambda x: isdir)
        with pytest.raises(argparse.ArgumentTypeError):
            valid_existing_dir_path("this/is/some/path")


class TestValidExistingPath:
    def test_existing_path_exists(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        result = valid_existing_path("this/is/some/path")

        assert type(result) is PosixPath

    def test_existing_path_not_exists(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: False)
        with pytest.raises(argparse.ArgumentTypeError):
            valid_existing_path("this/is/some/path")


class TestValidPlatforms:
    @pytest.mark.parametrize(
        "platforms",
        [
            ([Platform.IGXOrinDevIt]),
            ([Platform.JetsonAgxOrinDevKit]),
            ([Platform.X64Workstation]),
            ([Platform.IGXOrinDevIt, Platform.X64Workstation]),
            (
                [
                    Platform.IGXOrinDevIt,
                    Platform.X64Workstation,
                    Platform.JetsonAgxOrinDevKit,
                ]
            ),
        ],
    )
    def test_valid_platforms(self, platforms: list[Platform]):
        platform_strs = ",".join(x.value for x in platforms)
        result = valid_platforms(platform_strs)

        assert result == platforms

    @pytest.mark.parametrize(
        "platforms",
        [
            ("bad-platform"),
            (f"{Platform.IGXOrinDevIt.value},bad-platform"),
            (f"{Platform.IGXOrinDevIt.value},"),
        ],
    )
    def test_invalid_platforms(self, platforms: str):
        with pytest.raises(argparse.ArgumentTypeError):
            valid_platforms(platforms)


class TestValidPlatformConfiguration:
    @pytest.mark.parametrize(
        "platforms_config",
        [
            (PlatformConfiguration.dGPU.value),
            (PlatformConfiguration.iGPU.value),
        ],
    )
    def test_valid_platform_config(self, platforms_config: PlatformConfiguration):
        result = valid_platform_config(platforms_config)

        assert result.value == platforms_config

    @pytest.mark.parametrize(
        "platforms_config",
        [
            ("bad-platform-config"),
            (""),
        ],
    )
    def test_invalid_platform_config(self, platforms_config: str):
        with pytest.raises(argparse.ArgumentTypeError):
            valid_platform_config(platforms_config)


class TestValidSdkType:
    @pytest.mark.parametrize(
        "sdk_type",
        [
            (SdkType.Holoscan.value),
            (SdkType.MonaiDeploy.value),
        ],
    )
    def test_valid_sdk_type(self, sdk_type: SdkType):
        result = valid_sdk_type(sdk_type)

        assert result.value == sdk_type

    @pytest.mark.parametrize(
        "sdk_type",
        [
            ("bad-value"),
            (""),
        ],
    )
    def test_invalid_sdk_type(self, sdk_type: str):
        with pytest.raises(argparse.ArgumentTypeError):
            valid_sdk_type(sdk_type)
