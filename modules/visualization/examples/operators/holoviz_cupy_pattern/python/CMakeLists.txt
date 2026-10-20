# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT HOLOSCAN_BUILD_PYTHON)
    return()
endif()

set(_pattern_script "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_cupy_pattern.py")
install(FILES "${_pattern_script}"
    DESTINATION "${HOLOVIZ_EXAMPLES_INSTALL_DIR}/operators/holoviz_cupy_pattern/python"
    COMPONENT ${HOLOSCAN_VISUALIZATION_EXAMPLES_INSTALL_COMPONENT})

if(HOLOSCAN_BUILD_TESTS AND BUILD_TESTING)
    # Register independently of CuPy/device availability; only runtime probes may skip.
    add_test(NAME EXAMPLE_HOLOVIZ_CUPY_PATTERN.window
        COMMAND "${Python3_EXECUTABLE}" "${_pattern_script}" --seconds 1)
    set_tests_properties(EXAMPLE_HOLOVIZ_CUPY_PATTERN.window
        PROPERTIES ENVIRONMENT "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
            LABELS "examples;visualization;python;gpu" TIMEOUT 120)
    set_tests_properties(EXAMPLE_HOLOVIZ_CUPY_PATTERN.window PROPERTIES SKIP_RETURN_CODE 77)
endif()
unset(_pattern_script)
