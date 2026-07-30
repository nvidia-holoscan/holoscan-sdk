# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(PYTHON_DISTRIBUTED_TESTS
  test_application_exception_distributed.py
  test_distributed_app_implicit_broadcast.py
  test_distributed_app_three_ucx_receivers.py
  test_distributed_app_with_exception.py
  test_distributed_app_with_invalid_add_flow.py
  test_distributed_app_with_invalid_compose.py
  test_ucx_cpp_to_python.py
  test_ucx_message_serialization.py
)

# Exclude some tests from Multi-Threaded Scheduler coverage
# - omit UCX object serialization tests on this second run to keep overall time down
#   (estimated total EBS UCX test time: 2 minutes)
set(MTS_EXCLUDE_TEST_FILES
  test_distributed_app_three_ucx_receivers.py
  test_ucx_cpp_to_python.py
  test_ucx_message_serialization.py
)

# Distributed coverage with Event-Based Scheduler
holoscan_add_pytest_tests(
  FILES ${PYTHON_DISTRIBUTED_TESTS}
  TEST_DIRECTORY "${CMAKE_PYTHON_WORKING_DIR}/tests/system/distributed"
  WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  PREFIX python-api-system-distributed-ebs
  PYTEST_ARGS -v --durations=0
  ENVIRONMENT "${CMAKE_DISTRIBUTED_TEST_FLAGS};HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=event_based"
  RESOURCE_LOCK "distributed_ucx"
)

# Distributed coverage with Multi-Threaded Scheduler
set(MTS_DISTRIBUTED_TESTS ${PYTHON_DISTRIBUTED_TESTS})
list(REMOVE_ITEM MTS_DISTRIBUTED_TESTS ${MTS_EXCLUDE_TEST_FILES})
holoscan_add_pytest_tests(
  FILES ${MTS_DISTRIBUTED_TESTS}
  TEST_DIRECTORY "${CMAKE_PYTHON_WORKING_DIR}/tests/system/distributed"
  WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  PREFIX python-api-system-distributed-mts
  PYTEST_ARGS -v --durations=0 -k "not ucx"
  ENVIRONMENT "${CMAKE_DISTRIBUTED_TEST_FLAGS};HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=multi_thread"
  RESOURCE_LOCK "distributed_ucx"
)
