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

# Create examples
add_executable(ping_async
  ping_async.cpp
)
target_link_libraries(ping_async
  PUBLIC
  holoscan::core
  holoscan::ops::async_ping_rx
  holoscan::ops::async_ping_tx
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
)

# Copy config file to the build tree
add_custom_target(ping_async_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_async.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "ping_async.yaml"
    BYPRODUCTS "ping_async.yaml"
)
add_dependencies(ping_async ping_async_yaml)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(ping_async PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES ping_async.cpp
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
endif()

# Install the compiled example
install(TARGETS ping_async
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/ping_async.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  # Test all the combinations for the async test
  # RT: receive/transmit
  # R: receive only
  # T: transmit only
  # O: receive and transmit async set to false
  # M: Multi-thread
  # G: Greedy
  set(testconfigs RT T R O MRT GRT)
  foreach(config IN LISTS testconfigs)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/ping_async.yaml CONFIG_STRING)
    string(REPLACE "async_receive: true" "async_receive: false" CONFIG_STRING "${CONFIG_STRING}")

    string(FIND "${config}" "R" HAS_R)
    if(HAS_R GREATER -1)
      string(REPLACE "async_receive: false" "async_receive: true" CONFIG_STRING "${CONFIG_STRING}")
    endif()

    string(FIND "${config}" "T" HAS_T)
    if(HAS_T GREATER -1)
      string(REPLACE "async_transmit: false" "async_transmit: true" CONFIG_STRING "${CONFIG_STRING}")
    endif()

    string(FIND "${config}" "M" HAS_M)
    if(HAS_M GREATER -1)
      string(REPLACE "scheduler: event_based" "scheduler: multi_thread" CONFIG_STRING "${CONFIG_STRING}")
    endif()

    string(FIND "${config}" "G" HAS_G)
    if(HAS_G GREATER -1)
      string(REPLACE "scheduler: event_based" "scheduler: greedy" CONFIG_STRING "${CONFIG_STRING}")
    endif()

    # Write the config
    set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_video_replayer_config_${config}.yaml)
    file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

    add_test(NAME EXAMPLE_CPP_PING_ASYNC_${config}_TEST
             COMMAND ping_async ${CONFIG_FILE}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
    set_tests_properties(EXAMPLE_CPP_PING_ASYNC_${config}_TEST PROPERTIES
                         PASS_REGULAR_EXPRESSION "Rx message value: 20")
  endforeach()
endif()
