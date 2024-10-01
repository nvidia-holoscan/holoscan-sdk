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

import pytest

from holoscan.cli.common.constants import Constants, DefaultValues
from holoscan.cli.common.exceptions import InvalidSharedMemoryValueError
from holoscan.cli.runner.resources import _convert_to_bytes, get_shared_memory_size


class TestGetSharedMemorySize:
    def test_no_resources(self, monkeypatch):
        pkg_info = None
        result = get_shared_memory_size(pkg_info, False, False, None, "config")
        assert result == DefaultValues.DEFAULT_SHM_SIZE

        pkg_info = {}
        result = get_shared_memory_size(pkg_info, False, False, None, "config")
        assert result == DefaultValues.DEFAULT_SHM_SIZE

        pkg_info = {"not-a-resource": 0}
        result = get_shared_memory_size(pkg_info, False, False, None, "config")
        assert result == DefaultValues.DEFAULT_SHM_SIZE

    def test_no_global_shared_memory_value(self, monkeypatch):
        pkg_info = {"resources": {"memory": "1Gi"}}
        result = get_shared_memory_size(pkg_info, False, False, None, "config")
        assert result == DefaultValues.DEFAULT_SHM_SIZE

    def test_use_global_shared_memory_value(self, monkeypatch):
        pkg_info = {"resources": {Constants.RESOURCE_SHARED_MEMORY_KEY: "5Gi"}}
        result = get_shared_memory_size(pkg_info, False, False, None, "config")
        assert result == 5368709120

    def test_worker_with_resource_fragments_but_no_fragments_specified(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",
                "fragments": {
                    "fragment-a": {"sharedMemory": "5Mi"},
                    "fragment-b": {"sharedMemory": "0.5Gi"},  # expected
                    "fragment-c": {"memory": "5Gi"},
                },
            }
        }

        result = get_shared_memory_size(pkg_info, True, False, None, "config")
        assert result == 536870912

    def test_worker_with_resource_fragments_but_all_fragments_specified(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",
                "fragments": {
                    "fragment-a": {"sharedMemory": "5Mi"},
                    "fragment-b": {"sharedMemory": "0.5Gi"},  # expected
                    "fragment-c": {"memory": "5Gi"},
                },
            }
        }

        result = get_shared_memory_size(pkg_info, True, False, "all", "config")
        assert result == 536870912

    def test_worker_with_no_resource_fragments_but_no_fragments_specified(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",
            }
        }

        result = get_shared_memory_size(pkg_info, True, False, None, "config")
        assert result == 1048576

    def test_worker_with_no_resource_fragments_but_all_fragments_specified(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",
            }
        }

        result = get_shared_memory_size(pkg_info, True, False, "all", "config")
        assert result == 1048576

    def test_worker_with_resource_fragments_and_multiple_fragments_specified(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",
                "fragments": {
                    "fragment-a": {"sharedMemory": "5Mi"},  # expected
                    "fragment-b": {"sharedMemory": "0.5Gi"},
                    "fragment-c": {"memory": "5Gi"},
                },
            }
        }

        result = get_shared_memory_size(pkg_info, True, False, "fragment-a,fragment-c", "config")
        assert result == 5242880

    def test_worker_with_no_resource_fragments_and_multiple_fragments_specified(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",
            }
        }

        result = get_shared_memory_size(pkg_info, True, False, "fragment-a,fragment-c", "config")
        assert result == 1048576

    def test_driver_but_not_worker(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",  # expected
                "fragments": {
                    "fragment-a": {"sharedMemory": "5Mi"},
                    "fragment-b": {"sharedMemory": "0.5Gi"},
                    "fragment-c": {"memory": "5Gi"},
                },
            }
        }

        result = get_shared_memory_size(pkg_info, False, True, "fragment-a,fragment-c", "config")
        assert result == 1048576

    def test_not_driver_and_not_worker(self, monkeypatch):
        pkg_info = {
            "resources": {
                Constants.RESOURCE_SHARED_MEMORY_KEY: "1Mi",
                "fragments": {
                    "fragment-a": {"sharedMemory": "5Mi"},
                    "fragment-b": {"sharedMemory": "0.5Gi"},  # expected
                    "fragment-c": {"memory": "5Gi"},
                },
            }
        }

        result = get_shared_memory_size(pkg_info, False, False, "fragment-a,fragment-c", "config")
        assert result == 536870912

    @pytest.mark.parametrize(
        "value,expected_value",
        [
            (100, 100.0),
            (100.0, 100.0),
            ("5.5Mi", 5767168),
            ("5.5GB", 5500000000),
            ("5.5MiB", 5767168),
        ],
    )
    def test_user_values(self, value, expected_value):
        result = get_shared_memory_size([], False, False, "", value)
        assert result == expected_value


class TestConvertToBytes:
    def test_float_value(self, monkeypatch):
        result = _convert_to_bytes(100.0)
        assert result == 100.0

    def test_int_value(self, monkeypatch):
        result = _convert_to_bytes(100)
        assert result == 100

    def test_mebibytes(self, monkeypatch):
        result = _convert_to_bytes("5.5Mi")
        assert result == 5767168
        result = _convert_to_bytes("5.5 MiB")
        assert result == 5767168

    def test_gibibytes(self, monkeypatch):
        result = _convert_to_bytes("5.5Gi")
        assert result == 5905580032
        result = _convert_to_bytes("5.5 GiB")
        assert result == 5905580032

    def test_megabytes(self, monkeypatch):
        result = _convert_to_bytes("5.5 MB")
        assert result == 5500000
        result = _convert_to_bytes("5.5m")
        assert result == 5500000

    def test_gigabytes(self, monkeypatch):
        result = _convert_to_bytes("5.5GB")
        assert result == 5500000000
        result = _convert_to_bytes("5.5 g")
        assert result == 5500000000

    def test_unsupported_units(self, monkeypatch):
        with pytest.raises(InvalidSharedMemoryValueError):
            _convert_to_bytes("5.5 KB")
