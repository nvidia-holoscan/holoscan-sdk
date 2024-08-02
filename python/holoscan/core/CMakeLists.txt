# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

holoscan_pybind11_module(
    core
    application.cpp
    arg.cpp
    cli.cpp
    component.cpp
    condition.cpp
    core.cpp
    dataflow_tracker.cpp
    dl_converter.cpp
    emitter_receiver_registry.cpp
    execution_context.cpp
    executor.cpp
    fragment.cpp
    io_context.cpp
    io_spec.cpp
    kwarg_handling.cpp
    network_context.cpp
    operator.cpp
    resource.cpp
    scheduler.cpp
    tensor.cpp
    ../gxf/entity.cpp
)

# needed for the Holoviz InputSpec workaround in PyOutputContext::py_emit()
target_link_libraries(core_python
    PRIVATE holoscan::ops::holoviz
)

# Copy headers
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION include/holoscan/python
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "holoscan-python_libs"
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*pydoc.hpp" EXCLUDE
)
