# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_PYTHON_CORE_TEST_FILES
    test_hello.py
    test_tensor.py
)

set(_HOLOSCAN_PYTHON_TEST_PYTHONPATH "${CMAKE_BINARY_DIR}/python/lib")

holoscan_add_pytest_tests(
    FILES ${HOLOSCAN_PYTHON_CORE_TEST_FILES}
    TEST_DIRECTORY "${HOLOSCAN_PYTHON_TEST_BINARY_DIR}/core"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/python/lib"
    PREFIX python-core
    PYTEST_ARGS -v --durations=0
    ENVIRONMENT "PYTHONPATH=${_HOLOSCAN_PYTHON_TEST_PYTHONPATH}"
)
