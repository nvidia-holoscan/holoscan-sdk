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
import tempfile

import pytest
import yaml

from holoscan.cli.common.constants import DefaultValues, EnvironmentVariables
from holoscan.cli.common.enum_types import SdkType
from holoscan.cli.common.exceptions import InvalidApplicationConfigurationError
from holoscan.cli.packager.config_reader import ApplicationConfiguration
from holoscan.cli.packager.parameters import PackageBuildParameters


class TestApplicationConfiguration:
    def test_config_file_not_exists(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: False)
        config = ApplicationConfiguration()

        with pytest.raises(FileNotFoundError):
            config.read(pathlib.Path("/path/to/config/file"))

    def test_config_file_not_a_file(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "is_file", lambda x: False)
        config = ApplicationConfiguration()

        with pytest.raises(FileNotFoundError):
            config.read(pathlib.Path("/path/to/config/file"))

    def test_config_file_not_a_valid_extension(self, monkeypatch):
        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "is_file", lambda x: True)
        config = ApplicationConfiguration()

        with pytest.raises(InvalidApplicationConfigurationError):
            config.read(pathlib.Path("/path/to/config/file.json"))

    def test_config_file_open_exception(self, monkeypatch):
        def raise_error(self):
            raise Exception("error")

        monkeypatch.setattr(pathlib.Path, "exists", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "is_file", lambda x: True)
        monkeypatch.setattr(pathlib.Path, "open", raise_error)
        config = ApplicationConfiguration()

        with pytest.raises(InvalidApplicationConfigurationError):
            config.read(pathlib.Path("/path/to/config/file.yaml"))

    def test_config_yaml_load_exception(self, monkeypatch):
        data = self._generate_yaml()
        yaml_file = self._write_yaml(data)

        def raise_error(self, _):
            raise Exception("error")

        monkeypatch.setattr(yaml, "load", raise_error)
        config = ApplicationConfiguration()

        with pytest.raises(InvalidApplicationConfigurationError):
            config.read(pathlib.Path(yaml_file))

    def test_config_yaml_validate_ok(self, monkeypatch):
        data = self._generate_yaml()
        yaml_file = self._write_yaml(data)

        config = ApplicationConfiguration()

        config.read(pathlib.Path(yaml_file))

        assert config.title() == data["application"]["title"]

        pip_packages = config.pip_packages()
        assert pip_packages is not None
        assert len(pip_packages) == len(data["application"]["pip-packages"])
        for package in data["application"]["pip-packages"]:
            assert package in pip_packages

    def test_config_yaml_validate_ok_no_pip_packages(self, monkeypatch):
        data = self._generate_yaml()
        data["application"].pop("pip-packages")
        yaml_file = self._write_yaml(data)

        config = ApplicationConfiguration()

        config.read(pathlib.Path(yaml_file))

        assert config.title() == data["application"]["title"]

        pip_packages = config.pip_packages()
        assert pip_packages is None

    def test_config_yaml_validate_none(self, monkeypatch):
        yaml_file = self._write_yaml(self._generate_yaml())
        monkeypatch.setattr(yaml, "load", lambda x, y: None)

        config = ApplicationConfiguration()

        with pytest.raises(InvalidApplicationConfigurationError):
            config.read(pathlib.Path(yaml_file))

    def test_config_yaml_validate_missing_application(self, monkeypatch):
        data = self._generate_yaml()
        data.pop("application")
        yaml_file = self._write_yaml(data)

        config = ApplicationConfiguration()

        with pytest.raises(InvalidApplicationConfigurationError) as e:
            config.read(pathlib.Path(yaml_file))

        assert str(e).find("Application ('application') configuration cannot be found in ") != -1

    def test_config_yaml_validate_missing_resources(self, monkeypatch):
        data = self._generate_yaml()
        data.pop("resources")
        yaml_file = self._write_yaml(data)

        config = ApplicationConfiguration()

        with pytest.raises(InvalidApplicationConfigurationError) as e:
            config.read(pathlib.Path(yaml_file))

        assert str(e).find("Resources ('resources') configuration cannot be found in ") != -1

    def test_config_yaml_validate_missing_application_title(self, monkeypatch):
        data = self._generate_yaml()
        data["application"].pop("title")
        yaml_file = self._write_yaml(data)

        config = ApplicationConfiguration()

        with pytest.raises(InvalidApplicationConfigurationError) as e:
            config.read(pathlib.Path(yaml_file))

        assert str(e).find("Application configuration key/value ('application>title')") != -1

    def test_populate_app_manifest(self):
        data = self._generate_yaml()
        yaml_file = self._write_yaml(data)
        build_parameters = PackageBuildParameters()
        build_parameters._data["command"] = "/bin/bash my-command -and -args"

        config = ApplicationConfiguration()
        config.read(pathlib.Path(yaml_file))

        result = config.populate_app_manifest(build_parameters)
        assert result.api_version == DefaultValues.API_VERSION
        assert result.command == build_parameters.command
        assert (
            result.environment[EnvironmentVariables.HOLOSCAN_INPUT_PATH]
            == build_parameters.input_dir
        )
        assert (
            result.environment[EnvironmentVariables.HOLOSCAN_OUTPUT_PATH]
            == build_parameters.output_dir
        )
        assert result.environment[EnvironmentVariables.HOLOSCAN_WORKDIR] == str(
            build_parameters.working_dir
        )
        assert result.environment[EnvironmentVariables.HOLOSCAN_MODEL_PATH] == str(
            build_parameters.models_dir
        )
        assert result.environment[EnvironmentVariables.HOLOSCAN_CONFIG_PATH] == str(
            build_parameters.config_file_path
        )
        assert result.environment[EnvironmentVariables.HOLOSCAN_APP_MANIFEST_PATH] == str(
            build_parameters.app_manifest_path
        )
        assert result.environment[EnvironmentVariables.HOLOSCAN_PKG_MANIFEST_PATH] == str(
            build_parameters.package_manifest_path
        )
        assert result.input["path"] == build_parameters.input_dir
        assert result.input["formats"] == data["application"]["input-formats"]
        assert result.output["path"] == build_parameters.output_dir
        assert result.output["formats"] == data["application"]["output-formats"]
        assert result.readiness is None
        assert result.liveness is None
        assert result.timeout == build_parameters.timeout
        assert result.version == data["application"]["version"]
        assert result.working_directory == str(build_parameters.working_dir)

    def test_populate_app_manifest_use_version_in_build_parameters(self):
        data = self._generate_yaml()
        yaml_file = self._write_yaml(data)
        build_parameters = PackageBuildParameters()
        build_parameters._data["command"] = "/bin/bash my-command -and -args"
        build_parameters.version = "use this version"
        build_parameters.sdk = SdkType.Holoscan

        config = ApplicationConfiguration()
        config.read(pathlib.Path(yaml_file))

        result = config.populate_app_manifest(build_parameters)
        assert result.version == build_parameters.version

    def test_populate_app_manifest_raise_when_no_version_info(self):
        data = self._generate_yaml()
        data["application"].pop("version")
        yaml_file = self._write_yaml(data)
        build_parameters = PackageBuildParameters()
        build_parameters._data["command"] = "/bin/bash my-command -and -args"
        build_parameters.sdk = SdkType.Holoscan

        config = ApplicationConfiguration()
        config.read(pathlib.Path(yaml_file))

        with pytest.raises(InvalidApplicationConfigurationError) as e:
            config.populate_app_manifest(build_parameters)
        assert str(e).find("Application configuration key/value ('application>version')") != -1

    def test_populate_package_manifest_single_model(self):
        data = self._generate_yaml()
        yaml_file = self._write_yaml(data)
        build_parameters = PackageBuildParameters()
        build_parameters._data["command"] = "/bin/bash my-command -and -args"
        build_parameters.models = {"model-a": "/path/to/model"}
        config = ApplicationConfiguration()
        config.read(pathlib.Path(yaml_file))

        result = config.populate_package_manifest(build_parameters)

        assert result.api_version == DefaultValues.API_VERSION
        assert result.application_root == str(build_parameters.app_dir)
        assert result.model_root == str(build_parameters.models_dir)
        assert result.models["model-a"] == str(build_parameters.models_dir / "model-a")
        assert result.resources == data["resources"]
        assert result.version == data["application"]["version"]

    def test_populate_package_manifest_multiple_models(self):
        data = self._generate_yaml()
        yaml_file = self._write_yaml(data)
        build_parameters = PackageBuildParameters()
        build_parameters._data["command"] = "/bin/bash my-command -and -args"
        build_parameters.models = {"model-a": "/path/to/model-a", "model-b": "/path/to/model-b"}
        config = ApplicationConfiguration()
        config.read(pathlib.Path(yaml_file))

        result = config.populate_package_manifest(build_parameters)

        assert len(result.models) == len(build_parameters.models)
        for model in build_parameters.models:
            assert model in result.models
            assert result.models[model] == str(build_parameters.models_dir / model)

    def _generate_yaml(self) -> dict:
        data = {
            "application": {
                "title": "App Title",
                "version": "1.2.3",
                "fragments": "fragments",
                "input-formats": ["network"],
                "output-formats": ["screen"],
                "pip-packages": ["package-1", "package-2", "package-3"],
            },
            "resources": ["x", "y", "z"],
        }

        return data

    def _write_yaml(self, data) -> str:
        _, path = tempfile.mkstemp(suffix=".yaml")
        with open(path, mode="w") as file:
            file.write(yaml.dump(data))
        return path
