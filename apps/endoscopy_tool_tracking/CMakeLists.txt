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

add_executable(endoscopy_tool_tracking
    main.cpp
)

target_link_libraries(endoscopy_tool_tracking
   PRIVATE
   ${HOLOSCAN_PACKAGE_NAME}
)

# Download the associated dataset if needed
if(HOLOSCAN_DOWNLOAD_DATASETS)
  add_dependencies(endoscopy_tool_tracking endoscopy_data)
endif()

# Copy config file
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the app
install(TARGETS "endoscopy_tool_tracking"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-embedded-apps"
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml"
    DESTINATION ${app_relative_dest_path}
    COMPONENT "holoscan-embedded-apps"
)
