# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_executable(tensor_interop
  tensor_interop.cpp
  receive_tensor_gxf.hpp
  send_tensor_gxf.hpp
)
target_link_libraries(tensor_interop
  PRIVATE
  holoscan::core
  holoscan::data_loggers::basic_console_logger
  CUDA::cudart
)

# Install example

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(tensor_interop PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the target
install(TARGETS tensor_interop
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES tensor_interop.cpp receive_tensor_gxf.hpp send_tensor_gxf.hpp
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

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_TENSOR_INTEROP_TEST
    COMMAND tensor_interop
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_TENSOR_INTEROP_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Scheduling 3 entities"
    # data logging tests
    # note: tx and rx are GXFOperators (Codelets) so their ports are not logged
    PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:mx.in\\]"
    PASS_REGULAR_EXPRESSION "BasicConsoleLogger\\[ID:mx.out\\]"
    PASS_REGULAR_EXPRESSION "Category:TensorMap"
    PASS_REGULAR_EXPRESSION "TensorMap\\(size=1\\) \\{'tensor': Tensor\\(shape=\\[4, 4, 3], dtype=kDLUInt:8:1, device=kDLCUDA:0, size=48, nbytes=48, C-contiguous=true\\)\\}"
  )
endif()
