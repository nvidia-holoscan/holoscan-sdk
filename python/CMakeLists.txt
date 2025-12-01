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

# custom target to encapsulate all
add_custom_target(holoscan-python ALL
  COMMENT "adding holoscan-python target"
)

# python module
add_subdirectory(holoscan)

# For convenience, we currently copy the tests folder to build/python/lib as well

# custom target for Python unit test files
set(CMAKE_PYBIND11_TEST_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/tests)
set(CMAKE_PYTHON_WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}")
add_custom_target(holoscan-python-tests
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_PYBIND11_TEST_DIRECTORY}" "${CMAKE_PYTHON_WORKING_DIR}/tests"
  DEPENDS "${CMAKE_PYBIND11_CONFIG_FILE}"
  COMMENT "adding holoscan-python-tests target"
)
add_dependencies(holoscan-python holoscan-python-tests)

if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME python-api-unit-tests
    COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_PYTHON_WORKING_DIR}/tests/unit -v --durations=0
    # either have to run from this working directory or set PYTHONPATH
    WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  )
  set_tests_properties(python-api-unit-tests PROPERTIES
                       ENVIRONMENT "PYTHONUNBUFFERED=1")

  add_test(NAME python-api-system-tests
    COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_PYTHON_WORKING_DIR}/tests/system -v --durations=0 -k "not distributed" --run-realtime
    # either have to run from this working directory or set PYTHONPATH
    WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  )
  set_tests_properties(python-api-system-tests PROPERTIES
                       ENVIRONMENT "PYTHONUNBUFFERED=1")

  add_test(NAME python-api-system-distributed-tests
    COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_PYTHON_WORKING_DIR}/tests/system/distributed -v --durations=0
    # either have to run from this working directory or set PYTHONPATH
    WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  )

  # rerun distributed tests but with the event-based scheduler
  # (omit serialization tests on this second run to keep overall time down)
  add_test(NAME python-api-system-distributed-ebs-tests
    COMMAND ${PYTHON_EXECUTABLE} -m pytest ${CMAKE_PYTHON_WORKING_DIR}/tests/system/distributed -v --durations=0 -k "not test_ucx_object_serialization"
    # either have to run from this working directory or set PYTHONPATH
    WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  )

  # Tracking any fatal error reported by the application tests
  set_tests_properties(python-api-system-tests PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Fatal Python error")

  set_tests_properties(python-api-system-distributed-tests PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Fatal Python error")

  set_tests_properties(python-api-system-distributed-ebs-tests PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Fatal Python error")

# set environment variables used by distributed applications in the tests
# - UCX_TCP_CM_REUSEADDR=y : Reuse port for UCX
#   (see https://github.com/openucx/ucx/issues/8585 and https://github.com/rapidsai/ucxx#c-1)
# - HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=2500    : Set deadlock timeout for distributed app
# - HOLOSCAN_MAX_DURATION_MS=2500             : Set max duration for distributed app
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(CMAKE_DISTRIBUTED_TEST_FLAGS
"\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=6000;\
HOLOSCAN_MAX_DURATION_MS=6000\
"
  )
else()
  set(CMAKE_DISTRIBUTED_TEST_FLAGS
"\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=3000;\
HOLOSCAN_MAX_DURATION_MS=3000\
"
  )
endif()

  set_tests_properties(python-api-system-distributed-tests PROPERTIES ENVIRONMENT
    "${CMAKE_DISTRIBUTED_TEST_FLAGS} HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=multi_thread PYTHONUNBUFFERED=1"
  )
  set_tests_properties(python-api-system-distributed-ebs-tests PROPERTIES ENVIRONMENT
    "${CMAKE_DISTRIBUTED_TEST_FLAGS} HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=event_based PYTHONUNBUFFERED=1"
  )

  # tracing tests
  add_test(NAME python-api-tracing-profile
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_PYTHON_WORKING_DIR}/tests/system/test_pytracing.py profile
    # either have to run from this working directory or set PYTHONPATH
    WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  )
  set_tests_properties(python-api-tracing-profile PROPERTIES
                       FAIL_REGULAR_EXPRESSION "AssertionError"
                       SKIP_RETURN_CODE "12"
                       ENVIRONMENT "PYTHONUNBUFFERED=1")

  add_test(NAME python-api-tracing-trace
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_PYTHON_WORKING_DIR}/tests/system/test_pytracing.py trace
    # either have to run from this working directory or set PYTHONPATH
    WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  )
  set_tests_properties(python-api-tracing-trace PROPERTIES
                       FAIL_REGULAR_EXPRESSION "AssertionError"
                       SKIP_RETURN_CODE "12"
                       ENVIRONMENT "PYTHONUNBUFFERED=1")

  add_test(NAME python-api-tracing-coverage
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_PYTHON_WORKING_DIR}/tests/system/test_pytracing.py coverage
    # either have to run from this working directory or set PYTHONPATH
    WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  )
  set_tests_properties(python-api-tracing-coverage PROPERTIES
                       FAIL_REGULAR_EXPRESSION "AssertionError"
                       SKIP_RETURN_CODE "12"
                       ENVIRONMENT "PYTHONUNBUFFERED=1")
endif()
