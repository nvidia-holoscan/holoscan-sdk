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
"""This module provides library paths configurable via environment variables."""

import os

__all__ = ["holoscan_lib_path", "holoscan_gxf_extensions_path"]

current_file_directory = os.path.dirname(os.path.abspath(__file__))
holoscan_lib_env_path = os.environ.get("HOLOSCAN_LIB_PATH", "HOLOSCAN_LIB_PATH not set")
holoscan_install_relative_path = "../../../lib"
holoscan_wheel_relative_path = "lib"

potential_lib_paths = [
    os.path.realpath(holoscan_lib_env_path),
    os.path.realpath(os.path.join(current_file_directory, holoscan_install_relative_path)),
    os.path.realpath(os.path.join(current_file_directory, holoscan_wheel_relative_path)),
]

# raise an error if environment variables are set to a non-existent location
for lib_path in potential_lib_paths:
    if os.path.exists(lib_path):
        print(f"Found Holoscan C++ extension modules at {lib_path}")
        # Override the environment variable to ensure that the correct path is used by the framework
        # (by GXFExecutor::register_extensions() in the constructor of GXFExecutor class)
        os.environ["HOLOSCAN_LIB_PATH"] = lib_path
        holoscan_lib_path = lib_path
        holoscan_gxf_extensions_path = os.path.join(holoscan_lib_path, "gxf_extensions")
        break
else:
    raise ValueError(
        "Could not find the Holoscan C++ extension modules. Please specify the "
        "path to the libraries via the HOLOSCAN_LIB_PATH environment variable."
    )

del os
