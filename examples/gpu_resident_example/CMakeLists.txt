# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_CPP_EXAMPLES)
  # Create the GPU resident example executable
  add_executable(gpu_resident_example
    gpu_resident_example.cpp
    add_kernel.cu
  )
  target_link_libraries(gpu_resident_example
    PRIVATE
    holoscan::core
    holoscan::ops::ping_tx
    holoscan::ops::ping_rx
  )

  # Set the install RPATH based on the location of the Holoscan SDK libraries
  file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
  set_target_properties(gpu_resident_example PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

  if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    # Install the source
    install(FILES
      gpu_resident_example.cpp
      add_kernel.cu.hpp
      add_kernel.cu
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
  install(TARGETS gpu_resident_example
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
  )

  # Testing
  if(HOLOSCAN_BUILD_TESTS)
    add_test(NAME EXAMPLE_CPP_GPU_RESIDENT_TEST
             COMMAND gpu_resident_example
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
    set_tests_properties(EXAMPLE_CPP_GPU_RESIDENT_TEST PROPERTIES
                         PASS_REGULAR_EXPRESSION "All results are correct"
                         FAIL_REGULAR_EXPRESSION "Results are incorrect")
  endif()
endif()

install(
  FILES README.md
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
