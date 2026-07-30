# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
  include(HoloscanPytestAddTests)

  # Set environment variables used by distributed applications in the tests
  if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(CMAKE_DISTRIBUTED_TEST_FLAGS
"\
HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT=6000;\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=1000;\
HOLOSCAN_MAX_DURATION_MS=6000\
"
    )
  else()
    set(CMAKE_DISTRIBUTED_TEST_FLAGS
"\
HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT=3000;\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=1000;\
HOLOSCAN_MAX_DURATION_MS=3000\
"
    )
  endif()

  # Register individual pytest files as separate CTest entries
  add_subdirectory(tests/unit)
  add_subdirectory(tests/system)
  add_subdirectory(tests/system/distributed)

  # tracing tests
  # Holoscan SDK tracing is not yet fully implemented for Python 3.12 and above.
  # Tracking reference: 5144233
  if(Python3_VERSION VERSION_LESS "3.12")
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
endif()
