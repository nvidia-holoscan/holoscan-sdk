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

# custom target to copy __init__.py file
set(INIT_FILE_IN ${CMAKE_CURRENT_LIST_DIR}/__init__.py)
file(RELATIVE_PATH relative_dest_path ${CMAKE_CURRENT_BINARY_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR})
set(INIT_FILE_OUT ${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${relative_dest_path}/__init__.py)
add_custom_target(ping_tx_python
    COMMAND ${CMAKE_COMMAND} -E copy "${INIT_FILE_IN}" "${INIT_FILE_OUT}"
    DEPENDS "${INIT_FILE_IN}"
    BYPRODUCTS "${INIT_FILE_OUT}"
)
add_dependencies(holoscan-python ping_tx_python)
unset(INIT_FILE_IN)
unset(INIT_FILE_OUT)
