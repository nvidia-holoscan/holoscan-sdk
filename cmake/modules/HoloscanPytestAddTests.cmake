# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Register one CTest entry per pytest file.
# Follows the same pattern as ConfigureTest() in public/tests/CMakeLists.txt.
#
# Usage:
#   holoscan_add_pytest_tests(
#     FILES test_foo.py test_bar.py
#     TEST_DIRECTORY "${CMAKE_PYTHON_WORKING_DIR}/tests/unit"
#     WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
#     PREFIX python-api-unit
#     PYTEST_ARGS -v --durations=0
#     [TIMEOUT 1000]
#     [ENVIRONMENT "VAR1=val1;VAR2=val2"]
#   )
function(holoscan_add_pytest_tests)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(oneValueArgs TEST_DIRECTORY WORKING_DIRECTORY PREFIX TIMEOUT)
  set(multiValueArgs FILES PYTEST_ARGS ENVIRONMENT)
  cmake_parse_arguments(ARG "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(_file IN LISTS ARG_FILES)
    # e.g. test_conditions.py -> test_conditions
    string(REGEX REPLACE "\\.py$" "" _basename "${_file}")
    set(_test_name "${ARG_PREFIX}.${_basename}")

    add_test(NAME ${_test_name}
      COMMAND ${Python3_EXECUTABLE} -m pytest "${ARG_TEST_DIRECTORY}/${_file}"
              ${ARG_PYTEST_ARGS}
      WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY}
    )

    # Properties
    set(_env "PYTHONUNBUFFERED=1")
    if(ARG_ENVIRONMENT)
      list(APPEND _env ${ARG_ENVIRONMENT})
    endif()

    set_tests_properties(${_test_name} PROPERTIES
      ENVIRONMENT "${_env}"
      FAIL_REGULAR_EXPRESSION "Fatal Python error;nanobind: leaked"
    )

    if(ARG_TIMEOUT)
      set_tests_properties(${_test_name} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
    endif()
  endforeach()

endfunction()
