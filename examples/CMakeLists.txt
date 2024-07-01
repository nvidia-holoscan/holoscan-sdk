# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_subdirectory(aja_capture)
add_subdirectory(bring_your_own_model)
add_subdirectory(cli_packager)
add_subdirectory(conditions)
add_subdirectory(cupy_native)
add_subdirectory(flow_tracker)
add_subdirectory(hello_world)
add_subdirectory(holoviz)
add_subdirectory(import_gxf_components)
add_subdirectory(multithread)
add_subdirectory(multi_branch_pipeline)
add_subdirectory(numpy_native)
add_subdirectory(ping_any)
add_subdirectory(ping_conditional)
add_subdirectory(ping_cycle)
add_subdirectory(ping_simple)
add_subdirectory(ping_simple_run_async)
add_subdirectory(ping_custom_op)
add_subdirectory(ping_multi_port)
add_subdirectory(ping_distributed)
add_subdirectory(ping_vector)
add_subdirectory(python_decorator)
add_subdirectory(resources)
add_subdirectory(tensor_interop)
add_subdirectory(v4l2_camera)
add_subdirectory(video_replayer)
add_subdirectory(video_replayer_distributed)
add_subdirectory(wrap_operator_as_gxf_extension)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Generates the install CMakeLists.txt to compile all the cpp examples
set(cmake_example_directories)
file(GLOB_RECURSE cmakeinstallfiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} */CMakeLists.min.txt)
foreach(cmakeinstallfile IN LISTS cmakeinstallfiles)
  string(REPLACE "/CMakeLists.min.txt" "" cmakedirectory ${cmakeinstallfile})
  string(APPEND cmake_example_directories "add_subdirectory(${cmakedirectory})\n")
endforeach()

configure_file(CMakeLists.min.txt.in CMakeLists.min.txt @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.min.txt"
        RENAME "CMakeLists.txt"
        DESTINATION examples
        COMPONENT holoscan-examples
)

install(FILES README.md
        DESTINATION examples
        COMPONENT holoscan-examples
)

# Install CMake script to download example data from NGC
install(FILES ${CMAKE_SOURCE_DIR}/scripts/download_example_data
        DESTINATION examples
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        COMPONENT holoscan-examples
)

# Files for testing
install(FILES testing/run_example_tests
        DESTINATION examples/testing
        COMPONENT holoscan-examples
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/data/validation_frames
        DESTINATION examples/testing
        COMPONENT holoscan-examples
)

install(FILES ${CMAKE_SOURCE_DIR}/tests/recorder.hpp
        DESTINATION examples/testing
        COMPONENT holoscan-examples
)

endif()
