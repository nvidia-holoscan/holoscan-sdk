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

# Create library
add_library(ping_tx_native_op
  ping_tx_native_op.cpp
  ping_tx_native_op.hpp
)
target_link_libraries(ping_tx_native_op
  PUBLIC
    holoscan::core
)
set_target_properties(ping_tx_native_op
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Install operator

# Set the install RPATH based on the location of the Holoscan SDK libraries
file(RELATIVE_PATH install_lib_relative_path
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}
)
set_target_properties(ping_tx_native_op
  PROPERTIES
  INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}"
)

# Install following the relative folder path

file(RELATIVE_PATH relative_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES ping_tx_native_op.cpp ping_tx_native_op.hpp
  DESTINATION "${relative_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${relative_path}"
  COMPONENT holoscan-examples
)
endif()

# Install the compiled operator
install(TARGETS ping_tx_native_op
  DESTINATION "${relative_path}"
  COMPONENT "holoscan-examples"
)
