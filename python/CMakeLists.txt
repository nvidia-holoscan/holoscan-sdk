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

# create a folder to store the compiled libraries
set(CMAKE_PYBIND11_MODULE_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/holoscan)
file(MAKE_DIRECTORY ${CMAKE_PYBIND11_MODULE_OUT_DIR})

# custom target to encapsulate all
add_custom_target(holoscan-python ALL)

add_subdirectory(src)

# For convenience, we currently copy the tests folder to build/python/lib as well

# custom target for Python unit test files
set(CMAKE_PYBIND11_TEST_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests)
add_custom_target(holoscan-python-tests
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_PYBIND11_TEST_DIRECTORY}" "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/tests"
    DEPENDS "${CMAKE_PYBIND11_CONFIG_FILE}"
)
add_dependencies(holoscan-python holoscan-python-tests)

if(HOLOSCAN_BUILD_TESTS)
    add_test(NAME python-api-tests
        COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/tests
        # either have to run from this working directory or set PYTHONPATH
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    )
endif()
