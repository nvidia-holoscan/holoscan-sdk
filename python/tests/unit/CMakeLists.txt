# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(PYTHON_UNIT_TESTS
  test_conditions.py
  test_core.py
  test_core_metadata.py
  test_data_logger_import_dependency.py
  test_data_loggers.py
  test_decorator.py
  test_executors.py
  test_graphs.py
  test_gxf.py
  test_inference.py
  test_kwargs.py
  test_logger.py
  test_network_context.py
  test_operator_naming.py
  test_operators_import.py
  test_operators_native.py
  test_pose_tree.py
  test_resources.py
  test_schedulers.py
  test_subgraph_lifetime.py
)

holoscan_add_pytest_tests(
  FILES ${PYTHON_UNIT_TESTS}
  TEST_DIRECTORY "${CMAKE_PYTHON_WORKING_DIR}/tests/unit"
  WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  PREFIX python-api-unit
  PYTEST_ARGS -v --durations=0
)
