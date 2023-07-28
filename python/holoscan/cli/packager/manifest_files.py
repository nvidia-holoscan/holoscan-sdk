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

from pathlib import Path
from typing import Any, Dict


class ApplicationManifest:
    def __init__(self):
        self._data = {}
        self._data["apiVersion"] = None
        self._data["command"] = None
        self._data["environment"] = None
        self._data["input"] = None
        self._data["liveness"] = None
        self._data["output"] = None
        self._data["readiness"] = None
        self._data["sdk"] = None
        self._data["sdkVersion"] = None
        self._data["timeout"] = None
        self._data["version"] = None
        self._data["workingDirectory"] = None

    @property
    def api_version(self) -> str:
        return self._data["apiVersion"]

    @api_version.setter
    def api_version(self, value: str):
        self._data["apiVersion"] = value

    @property
    def command(self) -> str:
        return self._data["command"]

    @command.setter
    def command(self, value: str):
        self._data["command"] = value

    @property
    def environment(self) -> Dict[str, str]:
        return self._data["environment"]

    @environment.setter
    def environment(self, value: Dict[str, str]):
        self._data["environment"] = value

    @property
    def input(self) -> Dict[str, str]:
        return self._data["input"]

    @input.setter
    def input(self, value: Dict[str, str]):
        self._data["input"] = value

    @property
    def liveness(self) -> Any:
        return self._data["liveness"]

    @liveness.setter
    def liveness(self, value: Any):
        self._data["liveness"] = value

    @property
    def output(self) -> Dict[str, str]:
        return self._data["output"]

    @output.setter
    def output(self, value: Dict[str, str]):
        self._data["output"] = value

    @property
    def readiness(self) -> Any:
        return self._data["readiness"]

    @readiness.setter
    def readiness(self, value: Any):
        self._data["readiness"] = value

    @property
    def sdk(self) -> str:
        return self._data["sdk"]

    @sdk.setter
    def sdk(self, value: str):
        self._data["sdk"] = value

    @property
    def sdk_version(self) -> str:
        return self._data["sdkVersion"]

    @sdk_version.setter
    def sdk_version(self, value: str):
        self._data["sdkVersion"] = value

    @property
    def timeout(self) -> int:
        return self._data["timeout"]

    @timeout.setter
    def timeout(self, value: int):
        self._data["timeout"] = value

    @property
    def version(self) -> str:
        return self._data["version"]

    @version.setter
    def version(self, value: str):
        self._data["version"] = value

    @property
    def working_directory(self) -> str:
        return self._data["workingDirectory"]

    @working_directory.setter
    def working_directory(self, value: str):
        if isinstance(value, Path):
            self._data["workingDirectory"] = str(value)
        else:
            self._data["workingDirectory"] = value

    @property
    def data(self) -> Dict[str, Any]:
        """Returns all values for serializing to JSON"""
        return self._data


class PackageManifest:
    def __init__(self):
        self._data = {}
        self._data["apiVersion"] = None
        self._data["applicationRoot"] = None
        self._data["modelRoot"] = None
        self._data["models"] = None
        self._data["resources"] = None
        self._data["version"] = None

    @property
    def api_version(self) -> str:
        return self._data["apiVersion"]

    @api_version.setter
    def api_version(self, value: str):
        self._data["apiVersion"] = value

    @property
    def application_root(self) -> str:
        return self._data["applicationRoot"]

    @application_root.setter
    def application_root(self, value: str):
        if isinstance(value, Path):
            self._data["applicationRoot"] = str(value)
        else:
            self._data["applicationRoot"] = value

    @property
    def model_root(self) -> str:
        return self._data["modelRoot"]

    @model_root.setter
    def model_root(self, value: str):
        if isinstance(value, Path):
            self._data["modelRoot"] = str(value)
        else:
            self._data["modelRoot"] = value

    @property
    def models(self) -> Dict[str, str]:
        return self._data["models"]

    @models.setter
    def models(self, value: Dict[str, str]):
        self._data["models"] = value

    @property
    def resources(self) -> Any:
        return self._data["resources"]

    @resources.setter
    def resources(self, value: Any):
        """Resources are copied from application configuration file directly."""
        self._data["resources"] = value

    @property
    def version(self) -> str:
        return self._data["version"]

    @version.setter
    def version(self, value: str):
        self._data["version"] = value

    @property
    def data(self) -> Dict[str, Any]:
        """Returns all values for serializing to JSON"""
        return self._data
