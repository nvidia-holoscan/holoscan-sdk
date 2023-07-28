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

import logging
import re
from typing import Dict, Optional, Union

from ..common.constants import Constants, DefaultValues
from ..common.exceptions import InvalidSharedMemoryValue

logger = logging.getLogger("runner")


def get_shared_memory_size(
    pkg_info: dict, worker: bool, driver: bool, fragments: str, user_shm_size: Optional[str]
) -> Optional[float]:
    """Queries the pkg.json file for shared memory requirement of the application

    Args:
        pkg_info (dict): package manifest
        worker (bool): start the application as App Worker
        driver (bool): start the application as App Driver
        fragments (str): comma-separated list of fragments
        user_shm_size  (Optional[str]): user provided share memory size

    Returns:
        float: shared memory value in bytes
    """

    if user_shm_size == "config":
        return _read_shm_size_from_config(pkg_info, worker, driver, fragments)
    elif user_shm_size is not None:
        return _convert_to_bytes(user_shm_size)
    else:
        return None


def _read_shm_size_from_config(pkg_info: dict, worker: bool, driver: bool, fragments: str) -> float:
    """Queries the pkg.json file for shared memory requirement of the application

    Args:
        pkg_info (dict): package manifest
        worker (bool): start the application as App Worker
        driver (bool): start the application as App Driver
        fragments (str): comma-separated list of fragments

    Returns:
        float: shared memory value in bytes
    """
    resources = pkg_info.get("resources", None) if pkg_info is not None else None

    if resources is None:
        return DefaultValues.DEFAULT_SHM_SIZE

    max_value: float = 0
    global_shared_memory_size = _convert_to_bytes(
        resources.get(Constants.RESOURCE_SHARED_MEMORY_KEY, DefaultValues.DEFAULT_SHM_SIZE)
    )

    resources_fragments = resources.get("fragments", {})
    if worker:
        if fragments is None or fragments.lower() == "all":
            max_value = _find_maximum_shared_memory_value_from_all_fragments(resources_fragments)
        else:
            max_value = _find_maximum_shared_memory_value_from_matching_fragments(
                resources_fragments, fragments
            )
        max_value = max(max_value, global_shared_memory_size)
    else:
        if driver:
            max_value = global_shared_memory_size
        else:
            max_value = _find_maximum_shared_memory_value_from_all_fragments(resources_fragments)
            max_value = max(max_value, global_shared_memory_size)

    return max_value


def _find_maximum_shared_memory_value_from_matching_fragments(
    resources_fragments: Dict, fragments: str
) -> Optional[float]:
    """Scan matching fragments for the maximum shared memory value.

    Args:
        resources_fragments (Dict): fragment resources; resources->fragments
        fragments (str): comma-separated list of fragments to match

    Returns:
        Optional[float]: maximum shred memory value or zero if no fragments found.
    """
    user_fragments = fragments.split(",")
    fragment_values = [
        _convert_to_bytes(val[Constants.RESOURCE_SHARED_MEMORY_KEY])
        for key, val in resources_fragments.items()
        if Constants.RESOURCE_SHARED_MEMORY_KEY in val and key in user_fragments
    ]

    if fragment_values:
        return max(fragment_values)
    else:
        return 0


def _find_maximum_shared_memory_value_from_all_fragments(
    resources_fragments: Dict,
) -> Optional[float]:
    """Scan all fragments for the maximum shared memory value.

    Args:
        resources_fragments (Dict): fragment resources; resources->fragments

    Returns:
        Optional[float]: maximum shred memory value or zero if no fragments found.
    """
    fragment_values = [
        _convert_to_bytes(val[Constants.RESOURCE_SHARED_MEMORY_KEY])
        for _, val in resources_fragments.items()
        if Constants.RESOURCE_SHARED_MEMORY_KEY in val
    ]

    if fragment_values:
        return max(fragment_values)
    else:
        return 0


def _convert_to_bytes(raw_value: Union[str, float, int]) -> float:
    """Converts data measurements in string to float.

    Supported units are Mi|MiB (mebibytes), Gi|GiB (gibibytes), MB|m (megabytes) and GB|g (gigabytes)

    Args:
        raw_value (str): data measurement with a number and a supporting unit.

    Returns:
        float: number of bytes
    """

    if type(raw_value) in [float, int]:
        return raw_value

    result = re.search(r"([.\d]+)\s*(Mi|MiB|Gi|GiB|MB|GB|m|g)", raw_value)

    if result is not None:
        value = float(result.group(1))
        if result.group(2) in ["Mi", "MiB"]:
            return value * 1048576
        if result.group(2) in ["Gi", "GiB"]:
            return value * 1073741824
        if result.group(2) == "MB":
            return value * 1000000
        if result.group(2) == "GB":
            return value * 1000000000
        if result.group(2) == "m":
            return value * 1000000
        if result.group(2) == "g":
            return value * 1000000000

    raise InvalidSharedMemoryValue(f"Invalid/unsupported shared memory value: {raw_value}. ")
