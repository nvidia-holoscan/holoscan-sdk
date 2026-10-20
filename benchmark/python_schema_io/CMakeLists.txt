# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)

add_custom_target(python_schema_io_benchmark SOURCES schema_io_benchmark.py)
add_dependencies(python_schema_io_benchmark holoscan-python)
add_dependencies(holoscan-python-benchmark-targets python_schema_io_benchmark)

if(BUILD_TESTING)
    add_test(
        NAME BENCHMARK_PYTHON_SCHEMA_IO
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${HOLOSCAN_PYTHON_MODULE_ROOT}"
                ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/schema_io_benchmark.py"
                --iterations 64 --warmup 8 --repeats 1 --memory-iterations 16
    )
    set_tests_properties(BENCHMARK_PYTHON_SCHEMA_IO PROPERTIES LABELS benchmark TIMEOUT 120)
endif()
