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

# Create example
add_executable(aja_capture
  aja_capture.cpp
)

target_link_libraries(aja_capture
  PRIVATE
  holoscan::core
  holoscan::ops::aja
  holoscan::ops::holoviz
)

# Copy config file
add_custom_target(aja_capture_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "aja_capture.yaml"
  BYPRODUCTS "aja_capture.yaml"
)

add_dependencies(aja_capture aja_capture_yaml)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR} )
set_target_properties(aja_capture PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the app
install(TARGETS aja_capture
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.yaml"
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)
