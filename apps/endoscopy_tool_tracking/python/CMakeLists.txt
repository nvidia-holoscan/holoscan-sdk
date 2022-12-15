# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy endoscopy tool tracking application
add_custom_target(python_endoscopy_tool_tracking
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.py"
)
add_dependencies(${HOLOSCAN_PACKAGE_NAME} python_endoscopy_tool_tracking)

# Copy endoscopy tool tracking config file
add_custom_target(python_endoscopy_tool_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml"
)
add_dependencies(${HOLOSCAN_PACKAGE_NAME} python_endoscopy_tool_tracking_yaml)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-apps"
)
