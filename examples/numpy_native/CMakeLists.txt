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

# Copy native operator convolve application
add_custom_target(python_convolve ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/convolve.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "convolve.py"
  BYPRODUCTS "convolve.py"
)

# Install the app and doc
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/convolve.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_CONVOLVE_TEST
    COMMAND python3 convolve.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CONVOLVE_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "\[1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\]")
endif()
