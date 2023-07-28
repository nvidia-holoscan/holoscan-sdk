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

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy v4l2_camera application file
add_custom_target(python_v4l2_camera ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "v4l2_camera.py"
  BYPRODUCTS "v4l2_camera.py"
)

# Copy config file
add_custom_target(python_v4l2_camera_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "v4l2_camera.yaml"
  BYPRODUCTS "v4l2_camera.yaml"
)

add_dependencies(python_v4l2_camera python_v4l2_camera_yaml)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

