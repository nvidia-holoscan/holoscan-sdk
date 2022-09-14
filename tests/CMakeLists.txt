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

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest CMAKE_TEST_NAME)
  add_executable(${CMAKE_TEST_NAME} ${ARGN})
  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/gtests>"
  )

  target_include_directories(${CMAKE_TEST_NAME}
    PRIVATE
    ${HOLOSCAN_TOP}/src # in case we want to test the internal sources
  )

  target_link_libraries(${CMAKE_TEST_NAME}
    PRIVATE
    ${HOLOSCAN_PACKAGE_NAME}
    GTest::gmock_main
    GTest::gtest_main
  )

  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})

  install(
    TARGETS ${CMAKE_TEST_NAME}
    COMPONENT holoscan-embedded-testing
    DESTINATION bin/gtests/libholoscan-embedded
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# * core tests ----------------------------------------------------------------------------------
ConfigureTest(
  CORE_TEST
  core/config.cpp
  core/parameter.cpp
)

# #######
